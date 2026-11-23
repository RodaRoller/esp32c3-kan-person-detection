"""C3-02 step 4: verify the historical trained checkpoint (M-hist) and export its head as LUT v2.

Usage (from c3-02/):
    python scripts/checkpoint_head.py --kan-image-iot ../../upstream/kan-image-iot \
        --lut-kan ../../upstream/lut-kan [--archive-lut <package>/artifacts/lut_L16]

Input (private, git-ignored): ../private/checkpoint/kan_person_detector_best.pt
Outputs:
    ../private/c3-02/lut/layer{0..3}.npz          LUT v2 export of the trained head   (PRIVATE)
    results/checkpoint/checkpoint_summary.json    hashes, structure checks, sizes     (public)

Only metadata goes to results/: no weight, no LUT value, no per-image output.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

from build_reference import check_head_is_plain_sum, content_hash
from common import (CHECKPOINT, EXP01_MODEL_ARGS, LUT_CFG, PRIVATE_C302, REPO_DIR, RESULTS_DIR,
                    check_pinned, environment, f32, sha256, write_json)

ARRAYS = ("knots", "q_table", "scale", "y_min", "edge_base_scale", "edge_spline_scale", "edge_out_scale")


def load_checkpoint(path: Path, kan_image_iot: Path):
    """sha256 check, safe load (weights_only: no code executed), strict load into the public class."""
    digest = sha256(path)
    if digest != CHECKPOINT["sha256"]:
        sys.exit(f"checkpoint sha256 {digest} != expected {CHECKPOINT['sha256']}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sys.path.insert(0, str(kan_image_iot / "src"))
    from models.kan_model import KANImageClassifier
    # deep copy: pykan rewrites list arguments in place
    model = KANImageClassifier(**copy.deepcopy(EXP01_MODEL_ARGS))
    model.load_state_dict(ck["model_state_dict"], strict=True)   # raises on any missing/unexpected key
    model.eval()
    model.kan.kan.save_act = False
    return ck, model, digest


def count_params(model) -> dict:
    sd = model.state_dict()
    cnn = {k: v for k, v in sd.items() if k.startswith("preprocessor.")}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    t_cnn = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("preprocessor."))
    t_kan = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("kan."))
    # same counting as kan-image-iot KANImageClassifier (all nn.Parameters, trainable or not):
    # comparable with the counts recorded for the historical run (78,544 = CNN 43,976 + KAN 34,568)
    a_cnn = sum(p.numel() for p in model.preprocessor.parameters())
    a_kan = sum(p.numel() for p in model.kan.parameters())
    return {
        "all_params_total": a_cnn + a_kan, "all_params_cnn": a_cnn, "all_params_kan": a_kan,
        "trainable_total": t_cnn + t_kan, "trainable_cnn": t_cnn, "trainable_kan": t_kan,
        "cnn_conv_weight_shapes": {k: list(v.shape) for k, v in cnn.items() if v.dim() == 4},
        "cnn_conv_linear_weight_elements": int(sum(v.numel() for v in cnn.values() if v.dim() >= 2)),
        "cnn_state_tensors_float_elements": int(sum(v.numel() for v in cnn.values() if v.is_floating_point())),
        "kan_trainable_tensors": sorted({n.split(".")[-1] for n in trainable if n.startswith("kan.kan.act_fun")}),
    }


def grid_checks(pk) -> dict:
    out = {}
    for l, layer in enumerate(pk.act_fun):
        g = layer.grid.detach().numpy()
        out[f"layer{l}"] = {"one_grid_for_all_inputs": bool((g == g[0]).all()),
                            "x_min": f32(g[0, 0]), "x_max": f32(g[0, -1]), "points": int(g.shape[1])}
    return out


def compare_with_archive(mine: dict, archive_file: Path, load_lut_npz) -> dict:
    other = load_lut_npz(archive_file)
    res = {"archive_file": archive_file.name, "archive_file_sha256": sha256(archive_file), "arrays": {}}
    for name in ARRAYS:
        a, b = mine[name], getattr(other, name)
        same = a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        entry = {"identical": bool(same)}
        if not same and a is not None and b is not None and a.shape == b.shape:
            entry["max_abs_diff"] = f32(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        res["arrays"][name] = entry
    res["all_identical"] = all(v["identical"] for v in res["arrays"].values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kan-image-iot", type=Path, required=True)
    ap.add_argument("--lut-kan", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=REPO_DIR / CHECKPOINT["local_path"])
    ap.add_argument("--archive-lut", type=Path, default=None,
                    help="optional: artifacts/lut_L16 of the historical package, for a byte-level comparison")
    a = ap.parse_args()

    heads = {"kan-image-iot": check_pinned("kan-image-iot", a.kan_image_iot),
             "lut-kan": check_pinned("lut-kan", a.lut_kan)}
    torch.set_num_threads(1)

    ck, model, digest = load_checkpoint(a.checkpoint, a.kan_image_iot)
    pk = model.kan.kan
    plain_sum = check_head_is_plain_sum(pk)
    assert plain_sum["all_ok"], plain_sum
    grids = grid_checks(pk)
    assert all(g["one_grid_for_all_inputs"] for g in grids.values()), grids

    sys.path.insert(0, str(a.lut_kan))
    from src.models.kan_wrapper import PyKANSingleLayerAdapter
    from src.quant.lut_builder import artifact_memory_bytes, build_lut_for_edges
    from src.quant.lut_io import load_lut_npz, save_lut_npz

    lut_dir = PRIVATE_C302 / "lut"
    archive_files = sorted(a.archive_lut.glob("layer*.npz")) if a.archive_lut else []
    layers = []
    for l in range(len(pk.act_fun)):
        adapter = PyKANSingleLayerAdapter(pk, layer_idx=l)
        edges = adapter.extract_edges()
        art = build_lut_for_edges(edges, **LUT_CFG)
        path = lut_dir / f"layer{l}.npz"
        save_lut_npz(path, art)
        arrays = {n: getattr(art, n) for n in ARRAYS}
        entry = {"layer": l, "in_dim": adapter.in_dim, "out_dim": adapter.out_dim, "edges": len(edges),
                 "bytes_all_arrays": artifact_memory_bytes(art), "content_sha256": content_hash(arrays),
                 "private_file": str(path.relative_to(REPO_DIR)).replace("\\", "/")}
        if archive_files:
            entry["vs_archive"] = compare_with_archive(arrays, archive_files[l], load_lut_npz)
        layers.append(entry)

    exp = CHECKPOINT["expected"]
    summary = {
        "model_id": CHECKPOINT["id"],
        "checkpoint": {"file": CHECKPOINT["file"], "sha256": digest, "source": CHECKPOINT["source"],
                       "expected_local_path": CHECKPOINT["local_path"], "published_here": False},
        "contents": {"epoch": ck["epoch"], "val_acc_percent": f32(ck["val_acc"]), "used_ema": ck["used_ema"],
                     "batch_size": ck["model_info"]["config"]["training"]["batch_size"],
                     "keys": sorted(ck.keys())},
        "matches_expected": (ck["epoch"] == exp["epoch"] and ck["used_ema"] == exp["used_ema"]
                             and ck["model_info"]["config"]["training"]["batch_size"] == exp["batch_size"]),
        "load": {"loader": "torch.load(weights_only=True)",
                 "class": "kan-image-iot src/models/kan_model.py::KANImageClassifier",
                 "constructor_args": EXP01_MODEL_ARGS, "strict": True, "result": "all keys matched"},
        "params": count_params(model),
        "head_is_plain_sum": plain_sum,
        "grids": grids,
        "lut_config": LUT_CFG,
        "lut_layers": layers,
        "lut_total_bytes": sum(e["bytes_all_arrays"] for e in layers),
        "lut_identical_to_archive": (all(e["vs_archive"]["all_identical"] for e in layers)
                                     if archive_files else "not checked"),
        "upstream_commits": heads,
        "environment": environment(),
    }
    write_json(RESULTS_DIR / "checkpoint" / "checkpoint_summary.json", summary)
    print("sha256 ok, epoch", ck["epoch"], "used_ema", ck["used_ema"], "| params", summary["params"]["trainable_total"],
          "| LUT bytes", summary["lut_total_bytes"], "| identical to archive:", summary["lut_identical_to_archive"])


if __name__ == "__main__":
    main()
