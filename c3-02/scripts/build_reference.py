"""C3-02 step 2: fixture head -> LUT v2 export -> reference vectors.

Usage:
    python scripts/build_reference.py --lut-kan ../ext/lut-kan

Outputs (all under c3-02/):
    fixture/fixture_head_64-24-16-8-2_seed0.pt    pykan state_dict of the fixture head
    results/lut/layer{0..3}.npz                   lut-kan LUTArtifact v2, one file per KAN layer
    results/lut_manifest.json                     LUT parameters, shapes, sizes, hashes
    results/reference_vectors.json                inputs + expected outputs (pykan and LUT reference)

The fixture is pykan's deterministic initial state for the historical
architecture 64->24->16->8->2. It is NOT a trained model: the vectors test the
interface and the numerics of the engine, not person detection.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from common import (CLASS_NAMES, FIXTURE, FIXTURE_DIR, FIXTURE_NAME, LUT_CFG, RESULTS_DIR,
                    check_pinned, environment, f32, sha256, write_json)


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """Hash of names, dtypes, shapes and bytes: stable across runs, unlike .npz/.pt zip metadata."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build_fixture():
    from kan import KAN
    # pass a copy: pykan rewrites the width list in place ([64, ...] -> [[64, 0], ...])
    model = KAN(width=list(FIXTURE["width"]), grid=FIXTURE["grid"], k=FIXTURE["k"],
                seed=FIXTURE["seed"], auto_save=False)
    model.eval()
    return model


def check_head_is_plain_sum(model) -> dict:
    """The C++ engine will compute only y_j = sum_i phi_ij(x_i) per layer.
    Verify that pykan adds nothing else: identity node/subnode affines, empty symbolic branch."""
    checks = {}
    for l in range(len(model.act_fun)):
        checks[f"layer{l}"] = {
            "node_scale_is_1": bool(torch.all(model.node_scale[l] == 1)),
            "node_bias_is_0": bool(torch.all(model.node_bias[l] == 0)),
            "subnode_scale_is_1": bool(torch.all(model.subnode_scale[l] == 1)),
            "subnode_bias_is_0": bool(torch.all(model.subnode_bias[l] == 0)),
            "symbolic_mask_is_0": bool(torch.all(model.symbolic_fun[l].mask == 0)),
            "spline_mask_is_1": bool(torch.all(model.act_fun[l].mask == 1)),
        }
    checks["all_ok"] = all(all(v.values()) for k, v in checks.items() if k.startswith("layer"))
    return checks


def make_inputs(knots: np.ndarray, L: int) -> list[tuple[str, str, np.ndarray]]:
    """Head input vectors (float32[64]). The first ones hit exact contract corner cases on layer 0."""
    n = FIXTURE["width"][0]
    K = knots.size - 1
    idx = np.arange(n)
    alpha = np.linspace(0.0, 1.0, L, dtype=np.float32)
    grid = knots[:-1, None] + (knots[1:] - knots[:-1])[:, None] * alpha[None, :]   # [K, L]
    x_max, x_min = knots[-1], knots[0]
    above = np.nextafter(x_max, np.float32(np.inf), dtype=np.float32)
    below = np.nextafter(x_min, np.float32(-np.inf), dtype=np.float32)
    oob = np.array([above, below, 2.5, -2.5, 3.0, -3.0, 10.0, -10.0], dtype=np.float32)
    rng = [np.random.default_rng(s) for s in range(1, 6)]
    vecs = [
        ("V00_zeros", "all inputs 0", np.zeros(n)),
        ("V01_knots", "x_i = knot[i mod 12]: every knot incl. both domain ends (closed boundary)", knots[idx % (K + 1)]),
        ("V02_knots_shifted", "x_i = knot[(i+6) mod 12]: same corner cases on other inputs", knots[(idx + 6) % (K + 1)]),
        ("V03_lut_samples", "x_i = sample[k=i mod 11][l=5i mod 16]: exact LUT sample points (w = 0)", grid[idx % K, (5 * idx) % L]),
        ("V04_sample_midpoints", "midpoint of two consecutive samples (w = 0.5)",
         0.5 * (grid[idx % K, (3 * idx) % (L - 1)] + grid[idx % K, (3 * idx) % (L - 1) + 1])),
        ("V05_out_of_domain", "just outside both ends, then +-2.5, +-3, +-10: spline = 0, base branch kept", oob[idx % oob.size]),
        ("V06_relu_like_small", "|N(0, 0.5)|, seed 1: non-negative like the real post-ReLU features", np.abs(rng[0].normal(0, 0.5, n))),
        ("V07_relu_like_wide", "|N(0, 1.5)|, seed 2: non-negative, part of the inputs beyond 2.2", np.abs(rng[1].normal(0, 1.5, n))),
        ("V08_uniform_domain", "U(-2.2, 2.2), seed 3", rng[2].uniform(-2.2, 2.2, n)),
        ("V09_normal", "N(0, 1), seed 4", rng[3].normal(0, 1, n)),
        ("V10_uniform_positive", "U(0, 2.2), seed 5", rng[4].uniform(0, 2.2, n)),
        ("V11_constant_half", "all inputs 0.5", np.full(n, 0.5)),
    ]
    return [(i, d, np.asarray(v, dtype=np.float32)) for i, d, v in vecs]


def person_vectors(model, n_draws: int = 4000, seed: int = 7) -> list[tuple[str, str, np.ndarray]]:
    """All vectors above give 'no_person' on the untrained fixture. Pick, deterministically,
    the two U(-3, 3) draws with the largest positive margin z1 - z0, so the 'person'
    branch of the decision is exercised too. This is input selection, not training."""
    X = np.random.default_rng(seed).uniform(-3, 3, (n_draws, FIXTURE["width"][0])).astype(np.float32)
    with torch.no_grad():
        z = model(torch.from_numpy(X)).numpy()
    best = np.argsort(z[:, 1] - z[:, 0])[::-1][:2]
    return [(f"V{12 + r}_person_branch_{r + 1}",
             f"U(-3, 3) draw #{int(i)} of {n_draws} (seed {seed}) with rank-{r + 1} largest margin z1 - z0",
             X[i]) for r, i in enumerate(best)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-kan", type=Path, required=True)
    ap.add_argument("--regenerate-fixture", action="store_true",
                    help="overwrite fixture/*.pt with a new pykan initialisation (changes every expected value)")
    a = ap.parse_args()

    lut_kan_head = check_pinned("lut-kan", a.lut_kan)
    torch.set_num_threads(1)  # same summation order on every machine
    sys.path.insert(0, str(a.lut_kan))
    from src.kernels.lut_backend_reference import forward_reference
    from src.kernels.lut_contract import pack_dense_layer
    from src.models.kan_wrapper import PyKANSingleLayerAdapter
    from src.quant.lut_builder import artifact_memory_bytes, build_lut_for_edges
    from src.quant.lut_io import save_lut_npz
    from src.metrics.memory import lut_memory_report

    # 1. Fixture head. pykan's initialisation is not bit-reproducible (curve2coef uses a
    #    least-squares solve; two constructions differ by ~1 ULP in coef), so the fixture is
    #    defined by the committed file, not by the seed: create it once, then always load it.
    #    Loading uses strict=True (lut-kan's adapter from_arch() uses strict=False and
    #    swallows errors, which could silently leave random weights: not used here).
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = FIXTURE_DIR / f"{FIXTURE_NAME}.pt"
    if a.regenerate_fixture or not ckpt.exists():
        torch.save(build_fixture().state_dict(), ckpt)
        print("fixture created:", ckpt)
    model = build_fixture()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    model.save_act = False   # skip pykan's activation statistics (std over a batch of 1 warns); outputs unchanged
    sd_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    plain_sum = check_head_is_plain_sum(model)
    assert plain_sum["all_ok"], plain_sum

    # 2. One LUTArtifact v2 per KAN layer.
    lut_dir = RESULTS_DIR / "lut"
    packed_layers, layers_meta = [], []
    for l in range(len(model.act_fun)):
        adapter = PyKANSingleLayerAdapter(model, layer_idx=l)
        edges = adapter.extract_edges()
        art = build_lut_for_edges(edges, **LUT_CFG)
        path = lut_dir / f"layer{l}.npz"
        save_lut_npz(path, art)
        packed = pack_dense_layer(art, edges=edges, in_dim=adapter.in_dim, out_dim=adapter.out_dim,
                                  boundary_mode=LUT_CFG["boundary_mode"])
        packed_layers.append(packed)
        arrays = {"knots": art.knots, "q_table": art.q_table, "scale": art.scale, "y_min": art.y_min,
                  "edge_base_scale": art.edge_base_scale, "edge_spline_scale": art.edge_spline_scale,
                  "edge_out_scale": art.edge_out_scale}
        layers_meta.append({
            "layer": l, "file": f"results/lut/layer{l}.npz", "in_dim": adapter.in_dim, "out_dim": adapter.out_dim,
            "edges": len(edges), "edge_id": "out_idx * in_dim + in_idx",
            "K_segments": int(art.knots.size - 1), "L": art.L,
            "knots": [f32(t) for t in art.knots],
            "uniform_dx": packed.uniform_dx,  # None -> runtime uses the non-uniform (searchsorted) path
            "shapes": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
            "bytes_all_arrays": artifact_memory_bytes(art),
            "bytes_lut_memory_report": lut_memory_report(art)["lut_total_bytes"],  # excludes edge coefficients
            "content_sha256": content_hash(arrays),
            "file_sha256": sha256(path),
        })

    # 3. Reference vectors.
    knots0 = packed_layers[0].knots
    inputs = make_inputs(knots0, LUT_CFG["L"]) + person_vectors(model)
    X = np.stack([v for _, _, v in inputs]).astype(np.float32)
    # One vector at a time: torch's float32 reductions change in the last bits with the
    # batch size, and the expected values must not depend on which vectors are in the file.
    rows_pt = []
    with torch.no_grad():
        for x in X:
            xt = torch.from_numpy(x[None, :])
            out = model(xt).numpy()
            rows_pt.append([t.numpy()[0] for t in model.acts[1:]])   # h1, h2, h3, logits
            assert np.array_equal(out, rows_pt[-1][-1][None, :])
    acts_pt = [np.stack([r[i] for r in rows_pt]) for i in range(4)]

    acts_lut, h = [], X
    for packed in packed_layers:                               # chained: layer l eats LUT output of l-1
        h = forward_reference(h, packed)
        acts_lut.append(h)

    names = ["h1", "h2", "h3", "logits"]
    vectors = []
    for n, (vid, desc, x) in enumerate(inputs):
        zp, zl = acts_pt[-1][n], acts_lut[-1][n]
        vectors.append({
            "id": vid, "description": desc,
            "x": [f32(v) for v in x],
            "pykan": {nm: [f32(v) for v in acts_pt[i][n]] for i, nm in enumerate(names)},
            "lut_reference": {nm: [f32(v) for v in acts_lut[i][n]] for i, nm in enumerate(names)},
            "decision_pykan": CLASS_NAMES[int(np.argmax(zp))],
            "decision_lut_reference": CLASS_NAMES[int(np.argmax(zl))],
            "margin_z1_minus_z0": {"pykan": f32(zp[1] - zp[0]), "lut_reference": f32(zl[1] - zl[0])},
            "max_abs_diff_lut_vs_pykan": {nm: f32(np.abs(acts_lut[i][n] - acts_pt[i][n]).max())
                                          for i, nm in enumerate(names)},
        })

    summary = {
        "max_abs_diff_logits": f32(np.abs(acts_lut[-1] - acts_pt[-1]).max()),
        "max_abs_logit": f32(np.abs(acts_pt[-1]).max()),
        "max_abs_diff_per_layer": {nm: f32(np.abs(acts_lut[i] - acts_pt[i]).max()) for i, nm in enumerate(names)},
        "decisions_agree": sum(v["decision_pykan"] == v["decision_lut_reference"] for v in vectors),
        "n_vectors": len(vectors),
    }

    provenance = {
        "lut_kan_commit": lut_kan_head,
        "environment": environment(),
        "fixture": {**FIXTURE, "trained": False, "file": f"fixture/{FIXTURE_NAME}.pt",
                    "file_sha256": sha256(ckpt), "content_sha256": content_hash(sd_np),
                    "head_is_plain_sum": plain_sum},
        "lut_config": LUT_CFG,
    }
    write_json(RESULTS_DIR / "lut_manifest.json", {**provenance, "layers": layers_meta})
    write_json(RESULTS_DIR / "reference_vectors.json", {
        **provenance,
        "io_schema": {
            "input": "x: float32[64], head input (output of the CNN projection after ReLU in the real model)",
            "hidden": "h1: float32[24], h2: float32[16], h3: float32[8]",
            "output": "logits: float32[2], order " + str(CLASS_NAMES),
            "decision": "argmax(logits) == sign(z1 - z0); ties -> no_person",
            "pykan": "torch float32 forward of the fixture (expected float model)",
            "lut_reference": "lut-kan forward_reference, layers chained (expected LUT engine output)",
        },
        "summary": summary,
        "vectors": vectors,
    })

    print("layers:", [(m["in_dim"], m["out_dim"], m["edges"], m["bytes_all_arrays"], m["uniform_dx"]) for m in layers_meta])
    print("summary:", summary)
    print("fixture sha256:", provenance["fixture"]["file_sha256"])


if __name__ == "__main__":
    main()
