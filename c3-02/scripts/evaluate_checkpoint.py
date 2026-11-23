"""C3-02 step 5: run M-hist on the VWW validation and test images; compare pykan and LUT reference.

Usage (from c3-02/, after checkpoint_head.py):
    python scripts/evaluate_checkpoint.py --kan-image-iot ../../upstream/kan-image-iot \
        --lut-kan ../../upstream/lut-kan [--archive-results <package>/results]

The images are the vww_subset prepared by kan-image-iot (data/processed/vww_subset/{train,val,test}).
Outputs:
    ../private/c3-02/eval/{val,test}.npz           features, per-layer inputs, logits      (PRIVATE)
    results/checkpoint/test_evaluation.json        aggregate metrics only                  (public)

Two different questions are kept apart:
  * pykan vs LUT reference: how much the tables approximate the trained head (LUT design);
  * agreement with the archived scores: whether this run reproduces the historical one.
Accuracy figures are functional checks, not evidence of generalisation (see split_overlap).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from checkpoint_head import load_checkpoint
from common import (CHECKPOINT, CLASS_NAMES, EVAL_MEAN, EVAL_STD, HISTORICAL_THRESHOLD_P, LUT_CFG,
                    PRIVATE_C302, REPO_DIR, RESULTS_DIR, f32, write_json)


def image_loader(split_dir: Path, batch: int = 256):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                             transforms.Normalize(mean=EVAL_MEAN, std=EVAL_STD)])
    ds = datasets.ImageFolder(split_dir, transform=tf)
    assert ds.class_to_idx == {c: i for i, c in enumerate(CLASS_NAMES)}, ds.class_to_idx
    return ds, torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


@torch.no_grad()
def run_float_model(model, loader):
    """CNN features, then the pykan head layer by layer (inputs of every KAN layer are kept)."""
    pk = model.kan.kan
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.preprocessor(x).numpy())
        labels.append(y.numpy())
    X = np.concatenate(feats).astype(np.float32)
    layer_in, h = [], torch.from_numpy(X)
    for layer in pk.act_fun:              # valid because the head is a plain sum (checkpoint_head.py)
        layer_in.append(h.numpy().copy())
        h = layer(h)[0]
    logits = h.numpy()
    ref = pk(torch.from_numpy(X)).numpy()  # full pykan forward, for the record
    return X, np.concatenate(labels), layer_in, logits, float(np.abs(ref - logits).max())


def run_lut_reference(X, packed_layers, forward_reference):
    layer_in, h = [], X
    for p in packed_layers:
        layer_in.append(h)
        h = forward_reference(h, p)
    return layer_in, h


def confusion(y, pred) -> dict:
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    return {"accuracy": f32((tp + tn) / len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def domain_stats(layer_inputs, x_min: float, x_max: float) -> dict:
    out = {}
    for l, v in enumerate(layer_inputs):
        v = v.reshape(-1)
        out[f"layer{l}"] = {"values": int(v.size), "min": f32(v.min()), "max": f32(v.max()),
                            "below_domain_percent": f32(100 * np.mean(v < x_min)),
                            "above_domain_percent": f32(100 * np.mean(v > x_max)),
                            "vectors_with_any_input_outside": int(np.any((layer_inputs[l] < x_min) |
                                                                         (layer_inputs[l] > x_max), axis=1).sum())}
    return out


def split_overlap(data_dir: Path) -> dict:
    names = {s: {p.name for p in (data_dir / s).rglob("*.jpg")} for s in ("train", "val", "test")}
    return {"files": {s: len(v) for s, v in names.items()},
            "test_in_val": len(names["test"] & names["val"]), "test_in_train": len(names["test"] & names["train"]),
            "val_in_train": len(names["val"] & names["train"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kan-image-iot", type=Path, required=True)
    ap.add_argument("--lut-kan", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=REPO_DIR / CHECKPOINT["local_path"])
    ap.add_argument("--data", type=Path, default=None, help="vww_subset dir (default: <kan-image-iot>/data/processed/vww_subset)")
    ap.add_argument("--archive-results", type=Path, default=None, help="optional: results/ of the historical package")
    a = ap.parse_args()
    data_dir = a.data or a.kan_image_iot / "data" / "processed" / "vww_subset"

    torch.manual_seed(0)
    torch.set_num_threads(1)   
    _, model, _ = load_checkpoint(a.checkpoint, a.kan_image_iot)
    sys.path.insert(0, str(a.lut_kan))
    from src.kernels.lut_backend_reference import forward_reference
    from src.kernels.lut_contract import pack_dense_layer
    from src.models.kan_wrapper import PyKANSingleLayerAdapter
    from src.quant.lut_io import load_lut_npz

    pk = model.kan.kan
    packed = []
    for l in range(len(pk.act_fun)):
        ad = PyKANSingleLayerAdapter(pk, layer_idx=l)
        art = load_lut_npz(PRIVATE_C302 / "lut" / f"layer{l}.npz")
        packed.append(pack_dense_layer(art, edges=ad.extract_edges(), in_dim=ad.in_dim, out_dim=ad.out_dim,
                                       boundary_mode=LUT_CFG["boundary_mode"]))
    x_min, x_max = float(packed[0].x_min), float(packed[0].x_max)
    theta_hist = math.log(HISTORICAL_THRESHOLD_P / (1 - HISTORICAL_THRESHOLD_P))   

    out = {"model_id": CHECKPOINT["id"], "checkpoint_sha256": CHECKPOINT["sha256"],
           "data": "kan-image-iot data/processed/vww_subset, eval preprocessing: Resize(128,128), ToTensor, "
                   "Normalize(ImageNet mean/std)",
           "split_overlap": split_overlap(data_dir),
           "thresholds": {"argmax": {"rule": "person if z1 - z0 > 0 (ties -> no_person)"},
                          "historical": {"p_threshold": HISTORICAL_THRESHOLD_P, "rule_in_package": "p(person) >= t",
                                         "equivalent_margin_threshold": f32(theta_hist)}},
           "splits": {}}
    eval_dir = PRIVATE_C302 / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        ds, loader = image_loader(data_dir / split)
        X, y, pin, z_pk, pk_chain_diff = run_float_model(model, loader)
        lin, z_lut = run_lut_reference(X, packed, forward_reference)
        d_pk, d_lut = z_pk[:, 1] - z_pk[:, 0], z_lut[:, 1] - z_lut[:, 0]
        res = {"images": int(len(y)), "per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)},
               "layerwise_vs_full_pykan_max_abs": f32(pk_chain_diff)}
        # pykan vs LUT reference (approximation error of the tables on the trained head)
        res["lut_vs_pykan"] = {
            "logit_mae": f32(np.abs(z_lut - z_pk).mean()), "logit_max_abs": f32(np.abs(z_lut - z_pk).max()),
            "hidden_max_abs": {f"h{l}": f32(np.abs(lin[l] - pin[l]).max()) for l in range(1, 4)},
            "max_abs_logit": f32(np.abs(z_pk).max()),
            "decision_agreement_argmax": int(((d_pk > 0) == (d_lut > 0)).sum()),
            "decision_agreement_historical": int(((d_pk >= theta_hist) == (d_lut >= theta_hist)).sum()),
        }
        res["metrics"] = {
            "pykan_argmax": confusion(y, (d_pk > 0).astype(int)),
            "lut_argmax": confusion(y, (d_lut > 0).astype(int)),
            "pykan_historical_threshold": confusion(y, (d_pk >= theta_hist).astype(int)),
            "lut_historical_threshold": confusion(y, (d_lut >= theta_hist).astype(int)),
        }
        res["margins_lut"] = {
            "min_abs_margin": f32(np.abs(d_lut).min()),
            "count_abs_margin_below": {t: int((np.abs(d_lut) < float(t)).sum()) for t in ("1e-2", "1e-3", "1e-4")},
            "min_abs_distance_to_historical_threshold": f32(np.abs(d_lut - theta_hist).min()),
            "count_distance_to_historical_threshold_below": {t: int((np.abs(d_lut - theta_hist) < float(t)).sum())
                                                             for t in ("1e-2", "1e-3", "1e-4")},
            "exact_ties": int((z_lut[:, 0] == z_lut[:, 1]).sum()),
        }
        res["domain"] = {"x_min": f32(x_min), "x_max": f32(x_max), "pykan_chain": domain_stats(pin, x_min, x_max)}
        res["non_finite_outputs"] = int((~np.isfinite(z_lut)).sum() + (~np.isfinite(z_pk)).sum())

        if a.archive_results:
            fp = np.load(a.archive_results / f"EXP01_{'validation' if split == 'val' else 'test'}_FP32_scores.npz")
            cmp = {"labels_identical": bool(np.array_equal(fp["y_true"], y)),
                   "fp32_logit_max_abs_diff": f32(np.abs(fp["logits"] - z_pk).max()),
                   "fp32_argmax_agreement": int(((fp["logits"][:, 1] > fp["logits"][:, 0]) == (d_pk > 0)).sum())}
            lut_file = a.archive_results / f"EXP01_{split}_LUT8_scores.npz"
            if lut_file.exists():
                lu = np.load(lut_file)
                cmp["lut_logit_max_abs_diff"] = f32(np.abs(lu["logits"] - z_lut).max())
                cmp["lut_argmax_agreement"] = int(((lu["logits"][:, 1] > lu["logits"][:, 0]) == (d_lut > 0)).sum())
            res["vs_archived_scores"] = cmp

        np.savez_compressed(eval_dir / f"{split}.npz", features=X, labels=y, logits_pykan=z_pk, logits_lut=z_lut,
                            h1_lut=lin[1], h2_lut=lin[2], h3_lut=lin[3], files=np.array([p for p, _ in ds.samples]))
        out["splits"][split] = res
        print(split, res["lut_vs_pykan"], res["metrics"]["pykan_argmax"], res.get("vs_archived_scores"))

    write_json(RESULTS_DIR / "checkpoint" / "test_evaluation.json", out)
    print("overlap:", out["split_overlap"])


if __name__ == "__main__":
    main()
