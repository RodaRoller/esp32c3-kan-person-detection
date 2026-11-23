"""C3-02 step 6: numeric acceptance criterion (measured) and reference vectors for M-hist.

Usage (from c3-02/, after evaluate_checkpoint.py):
    python scripts/tolerance_and_vectors.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan

What it does
1. A float32 model of the future C++ loop (i outer, j inner, one SiLU per input, float32 everywhere,
   FP16 metadata widened to float32). It is a measuring tool written in numpy, not the engine.
   It is run in "legitimate" variants (what a correct C++ may differ in: libm expf, reciprocals
   instead of divisions, SiLU formula) and in "faulty" variants (contract violations), against the
   lut-kan forward_reference. The tolerance must accept the first group and reject the second.
2. Reference vectors on the trained head: real feature vectors chosen for the cases that matter,
   the synthetic corner cases of the F0 set, and non-finite inputs.

Outputs:
    results/checkpoint/tolerance.json                    measured differences, proposal  (public)
    ../private/c3-02/reference_vectors_mhist.json        vectors with expected outputs   (PRIVATE)
    results/checkpoint/reference_vectors_mhist_index.json ids, cases, hash of the private file (public)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from build_reference import make_inputs
from checkpoint_head import load_checkpoint
from common import (CHECKPOINT, CLASS_NAMES, HISTORICAL_THRESHOLD_P, LUT_CFG, PRIVATE_C302, REPO_DIR,
                    RESULTS_DIR, f32, sha256, write_json)

F = np.float32
LEGIT = ["replica", "expf_1ulp", "reciprocal_dx", "silu_via_sigmoid", "all_legit_combined"]
# differs from the contract text, but picks the same segment at all 12 float32 knots of this grid:
EQUIVALENT_HERE = ["uniform_segment_formula"]
FAULTY = ["cell_lookup_no_interp", "no_oob_zero", "half_open_boundary", "metadata_as_float16_math"]


def perturb_ulp(v: np.ndarray, rng) -> np.ndarray:
    """Move every value by -1, 0 or +1 ULP: a model of a different (correctly rounded within 1 ULP) expf."""
    steps = rng.integers(-1, 2, v.shape)
    out = v.copy()
    out[steps > 0] = np.nextafter(v[steps > 0], F(np.inf))
    out[steps < 0] = np.nextafter(v[steps < 0], F(-np.inf))
    return out


def layer_f32(X: np.ndarray, p, variant: str, rng) -> np.ndarray:
    """One KAN layer, contract v2, float32, C loop order. X: [N, in] float32 -> [N, out] float32."""
    N, n_in = X.shape
    n_out = p.q_flat.shape[1]
    K, L = int(p.K), int(p.L)
    knots = np.asarray(p.knots, dtype=F)
    x_min, x_max = F(p.x_min), F(p.x_max)
    legit = variant in ("expf_1ulp", "reciprocal_dx", "silu_via_sigmoid", "all_legit_combined")
    y = np.zeros((N, n_out), dtype=F)
    inv_d = (F(1.0) / (knots[1:] - knots[:-1])).astype(F)
    dx_uniform = F((x_max - x_min) / F(K))
    for i in range(n_in):
        x = X[:, i].astype(F)
        e = np.exp(-x, dtype=F)
        if variant in ("expf_1ulp", "all_legit_combined"):
            e = perturb_ulp(e, rng)
        if variant in ("silu_via_sigmoid", "all_legit_combined"):
            base = (x * (F(1.0) / (F(1.0) + e))).astype(F)
        else:
            base = (x / (F(1.0) + e)).astype(F)
        if variant == "half_open_boundary":
            in_dom = (x >= x_min) & (x < x_max)
            xc = np.clip(x, x_min, np.nextafter(x_max, F(-np.inf)))
        else:
            in_dom = (x >= x_min) & (x <= x_max)
            xc = np.clip(x, x_min, x_max)
        if variant == "uniform_segment_formula":              # t_k = x_min + k*dx, as in lut-kan mcu_auto / IDS
            k = np.clip(np.floor((xc - x_min) / dx_uniform).astype(np.int64), 0, K - 1)
            u = np.clip((xc - (x_min + k.astype(F) * dx_uniform)) / dx_uniform, F(0), F(1)).astype(F)
        else:
            k = np.clip(np.searchsorted(knots, xc, side="right") - 1, 0, K - 1)
            if variant in ("reciprocal_dx", "all_legit_combined"):
                u = ((xc - knots[k]) * inv_d[k]).astype(F)
            else:
                u = ((xc - knots[k]) / np.maximum(knots[k + 1] - knots[k], F(1e-12))).astype(F)
            u = np.clip(u, F(0), F(1))
        pos = (u * F(L - 1)).astype(F)
        r0 = np.clip(np.floor(pos).astype(np.int64), 0, L - 1)
        r1 = np.clip(r0 + 1, 0, L - 1)
        w = (pos - r0.astype(F)).astype(F)
        q0 = p.q_flat[i][:, k * L + r0].T.astype(F)              # [N, out]
        q1 = p.q_flat[i][:, k * L + r1].T.astype(F)
        if variant == "metadata_as_float16_math":             # arithmetic done in float16 (wrong on purpose)
            ym = p.y_min[i][:, k].T.astype(np.float16); sc = p.scale[i][:, k].T.astype(np.float16)
            y0 = (ym + sc * q0.astype(np.float16)).astype(F); y1 = (ym + sc * q1.astype(np.float16)).astype(F)
        else:
            ym = p.y_min[i][:, k].T.astype(F); sc = p.scale[i][:, k].T.astype(F)
            y0 = (ym + sc * q0).astype(F); y1 = (ym + sc * q1).astype(F)
        if variant == "cell_lookup_no_interp":                # IDS-style: value of the cell, no interpolation
            s = y0
        else:
            s = ((F(1.0) - w)[:, None] * y0 + w[:, None] * y1).astype(F)
        if variant != "no_oob_zero":
            s = (s * in_dom[:, None].astype(F)).astype(F)
        t = (p.coef_out[i][None, :].astype(F) * (p.coef_base[i][None, :].astype(F) * base[:, None]
                                                 + p.coef_lut[i][None, :].astype(F) * s)).astype(F)
        y = (y + t).astype(F)
    return y


def chain(X, packed, fn):
    acts, h = [], X.astype(F)
    for p in packed:
        h = fn(h, p)
        acts.append(h)
    return acts          # [h1, h2, h3, logits]


def rel_err(a, b) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b) / np.maximum(1.0, np.abs(b.astype(np.float64)))))


def knot_jumps(packed) -> dict:
    """Discontinuity of the LUT at every interior knot (value from the left segment vs from the right one),
    weighted by the edge coefficients: what a wrong segment choice at a knot would cost on one edge."""
    out = {}
    for l, p in enumerate(packed):
        K, L = int(p.K), int(p.L)
        q = p.q_flat.reshape(p.q_flat.shape[0], p.q_flat.shape[1], K, L).astype(np.float64)
        ym, sc = p.y_min.astype(np.float64), p.scale.astype(np.float64)
        left = ym[..., :-1] + sc[..., :-1] * q[..., :-1, L - 1]
        right = ym[..., 1:] + sc[..., 1:] * q[..., 1:, 0]
        jump = np.abs(left - right) * np.abs(p.coef_out * p.coef_lut)[..., None]
        out[f"layer{l}"] = {"median": f32(np.median(jump)), "p90": f32(np.percentile(jump, 90)),
                            "max": f32(jump.max()), "share_above_1e-5": f32(np.mean(jump > 1e-5))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kan-image-iot", type=Path, required=True)
    ap.add_argument("--lut-kan", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=REPO_DIR / CHECKPOINT["local_path"])
    a = ap.parse_args()
    torch.set_num_threads(1)

    _, model, _ = load_checkpoint(a.checkpoint, a.kan_image_iot)
    pk = model.kan.kan
    sys.path.insert(0, str(a.lut_kan))
    from src.kernels.lut_backend_reference import forward_reference
    from src.kernels.lut_contract import pack_dense_layer
    from src.models.kan_wrapper import PyKANSingleLayerAdapter
    from src.quant.lut_io import load_lut_npz
    packed = []
    for l in range(len(pk.act_fun)):
        ad = PyKANSingleLayerAdapter(pk, layer_idx=l)
        packed.append(pack_dense_layer(load_lut_npz(PRIVATE_C302 / "lut" / f"layer{l}.npz"), edges=ad.extract_edges(),
                                       in_dim=ad.in_dim, out_dim=ad.out_dim, boundary_mode=LUT_CFG["boundary_mode"]))

    # inputs: every validation feature vector (the 4,000 test images are a subset of them) + synthetic corner cases
    val = np.load(PRIVATE_C302 / "eval" / "val.npz")
    synth = make_inputs(np.asarray(packed[0].knots, dtype=F), LUT_CFG["L"])
    X_syn = np.stack([v for _, _, v in synth]).astype(F)
    X = np.concatenate([val["features"].astype(F), X_syn])
    theta_hist = math.log(HISTORICAL_THRESHOLD_P / (1 - HISTORICAL_THRESHOLD_P))

    ref = chain(X, packed, forward_reference)
    d_ref = ref[-1][:, 1] - ref[-1][:, 0]
    names = ["h1", "h2", "h3", "logits"]
    variants = {}
    for v in LEGIT + EQUIVALENT_HERE + FAULTY:
        rng = np.random.default_rng(12345)
        acts = chain(X, packed, lambda h, p: layer_f32(h, p, v, rng))
        d = acts[-1][:, 1] - acts[-1][:, 0]
        variants[v] = {
            "kind": ("legitimate" if v in LEGIT else
                     "deviation, same segments on this grid" if v in EQUIVALENT_HERE else "contract violation"),
            "max_abs": {n: f32(np.abs(acts[i].astype(np.float64) - ref[i]).max()) for i, n in enumerate(names)},
            "max_rel_vs_max1": {n: f32(rel_err(acts[i], ref[i])) for i, n in enumerate(names)},
            "decision_mismatches_argmax": int(((d > 0) != (d_ref > 0)).sum()),
            "decision_mismatches_historical": int(((d >= theta_hist) != (d_ref >= theta_hist)).sum()),
        }
        worst = np.max(np.abs(np.concatenate([x.astype(np.float64) - r for x, r in zip(acts, ref)], 1))
                       / np.maximum(1.0, np.abs(np.concatenate(ref, 1))), axis=1)          # per vector, all layers
        variants[v]["vectors_failing_tolerance"] = {t: int((worst > float(t)).sum()) for t in ("1e-6", "2e-6", "1e-5")}
        print(f"{v:28s} max_rel {max(variants[v]['max_rel_vs_max1'].values()):.3e} failing {variants[v]['vectors_failing_tolerance']}")
    legit_max = max(max(variants[v]["max_rel_vs_max1"].values()) for v in LEGIT)
    faulty_min = min(max(variants[v]["max_rel_vs_max1"].values()) for v in FAULTY)

    lut_err = json.loads((RESULTS_DIR / "checkpoint" / "test_evaluation.json").read_text())["splits"]["val"]
    write_json(RESULTS_DIR / "checkpoint" / "tolerance.json", {
        "model_id": CHECKPOINT["id"],
        "inputs": {"validation_feature_vectors": int(len(val["features"])), "synthetic_vectors": int(len(X_syn)),
                   "note": "the 4,000 test images are a subset of the 4,830 validation images"},
        "metric": "|engine - lut_reference| / max(1, |lut_reference|), on h1, h2, h3 and logits",
        "variants": variants,
        "largest_legitimate_difference": f32(legit_max),
        "smallest_violation_difference": f32(faulty_min),
        "lut_vs_pykan_logit_max_abs": lut_err["lut_vs_pykan"]["logit_max_abs"],
        "smallest_abs_margin_lut_reference_validation": lut_err["margins_lut"]["min_abs_margin"],
        "knot_discontinuity_weighted": knot_jumps(packed),
    })

    # ---- reference vectors (private) ----
    idx_real = len(val["features"])
    pin = [X] + ref[:-1]
    z = ref[-1][:idx_real]; y = val["labels"]; d = d_ref[:idx_real]
    out0 = np.sum(val["features"] > packed[0].x_max, axis=1)
    h1 = ref[0][:idx_real]
    h3 = ref[2][:idx_real]
    out3 = np.sum((h3 < packed[3].x_min) | (h3 > packed[3].x_max), axis=1)
    pick = [
        ("R01_confident_person", "largest margin z1 - z0", int(np.argmax(d))),
        ("R02_confident_no_person", "smallest margin z1 - z0", int(np.argmin(d))),
        ("R03_near_argmax_above", "smallest positive margin (closest to theta = 0)", int(np.where(d > 0, d, np.inf).argmin())),
        ("R04_near_argmax_below", "largest negative margin (closest to theta = 0)", int(np.where(d <= 0, d, -np.inf).argmax())),
        ("R05_near_historical_above", "closest to the historical threshold, above", int(np.where(d >= theta_hist, d, np.inf).argmin())),
        ("R06_near_historical_below", "closest to the historical threshold, below", int(np.where(d < theta_hist, d, -np.inf).argmax())),
        ("R07_most_inputs_out_of_domain_l0", "most layer-0 inputs above 2.2", int(np.argmax(out0))),
        ("R08_most_negative_h1", "most negative layer-1 input (deep out of domain)", int(np.argmin(h1.min(axis=1)))),
        ("R09_most_out_of_domain_l3", "most layer-3 inputs outside [-2.2, 2.2]", int(np.argmax(out3))),
        ("R10_false_negative", "person image classified no_person with the largest error",
         int(np.where(y == 1, d, np.inf).argmin())),
    ]
    used, unique = set(), []
    for vid, desc, n in pick:
        if n in used:
            order = {"R08_most_negative_h1": np.argsort(h1.min(axis=1)),
                     "R09_most_out_of_domain_l3": np.argsort(-out3)}.get(vid, np.argsort(-out0))
            n = int(next(m for m in order if int(m) not in used))
            desc += " (next best: the top image is already used)"
        used.add(n)
        unique.append((vid, desc, n))
    pick = unique
    vecs = []
    def entry(vid, desc, x, acts_ref, pk_out, source):
        zz = acts_ref[-1]
        dd = float(zz[1]) - float(zz[0]) if np.all(np.isfinite(zz)) else float("nan")
        return {"id": vid, "description": desc, "source": source, "x": [f32(v) for v in x],
                "lut_reference": {n: [f32(v) for v in acts_ref[i]] for i, n in enumerate(names)},
                "pykan_logits": [f32(v) for v in pk_out] if pk_out is not None else None,
                "margin_z1_minus_z0": f32(dd),
                "decision_argmax": CLASS_NAMES[int(dd > 0)] if math.isfinite(dd) else "reject (non-finite)",
                "decision_historical_threshold": CLASS_NAMES[int(dd >= theta_hist)] if math.isfinite(dd) else "reject (non-finite)"}
    for vid, desc, n in pick:
        vecs.append(entry(vid, desc + f" (label {CLASS_NAMES[int(y[n])]})", X[n], [r[n] for r in ref],
                          val["logits_pykan"][n], "validation image " + Path(str(val["files"][n])).name))
    with torch.no_grad():
        for s, (vid, desc, x) in enumerate(synth[:6]):
            vecs.append(entry("S" + vid[1:], desc, x, [r[idx_real + s] for r in ref],
                              pk(torch.from_numpy(x[None, :])).numpy()[0], "synthetic (same rule as the F0 set)"))
    base = val["features"][int(np.argmax(d))].astype(F)
    for vid, desc, bad in (("N01_nan", "input 0 = NaN, others from R01", np.nan),
                           ("N02_pos_inf", "input 0 = +Inf, others from R01", np.inf),
                           ("N03_neg_inf", "input 0 = -Inf, others from R01", -np.inf)):
        x = base.copy(); x[0] = F(bad)
        with np.errstate(all="ignore"):
            acts = chain(x[None, :], packed, forward_reference)
        vecs.append(entry(vid, desc + " -> reference output recorded; required C++ behaviour: reject the input",
                          x, [r_[0] for r_ in acts], None, "synthetic"))
    priv = PRIVATE_C302 / "reference_vectors_mhist.json"
    write_json(priv, {"model_id": CHECKPOINT["id"], "checkpoint_sha256": CHECKPOINT["sha256"],
                      "historical_margin_threshold": f32(theta_hist), "vectors": vecs})
    write_json(RESULTS_DIR / "checkpoint" / "reference_vectors_mhist_index.json", {
        "model_id": CHECKPOINT["id"], "private_file": str(priv.relative_to(REPO_DIR)).replace("\\", "/"),
        "private_file_sha256": sha256(priv), "published_here": False,
        "vectors": [{"id": v["id"], "description": v["description"], "source": v["source"].split(" COCO")[0]}
                    for v in vecs]})
    print("legit max", legit_max, "| faulty min", faulty_min, "| vectors", len(vecs))


if __name__ == "__main__":
    main()
