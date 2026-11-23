"""C3-02 step 3: analytic Flash and RAM estimates, head only and full pipeline.

Usage:
    python scripts/estimate_memory.py          # needs results/lut_manifest.json from build_reference.py

Writes results/memory_estimate.json and results/memory_estimate.md.
Every figure carries a status:
  computed  - exact from array shapes and data types (checked against the built artifact)
  estimated - depends on implementation choices not made yet
  to measure - only the build/run can tell (pio run -t size, ESP.getFreeHeap(), stack high-water mark)
The CNN figures are analytical scenarios, not a demonstrated feasibility on the MCU.
"""
from __future__ import annotations

import json

from common import LUT_CFG, RESULTS_DIR, write_json

K_SEGMENTS = 11          # pykan grid=5, k=3 -> 12 knots -> 11 segments
F32, F16, U8, I32 = 4, 2, 1, 4


HEADS = [
    ("H0", [64, 24, 16, 8, 2], "historical run; trained weights = M-hist (private); same shape as the fixture F0"),
    ("S1", [64, 16, 8, 2], "smaller head, same 64-d features (would need training)"),
    ("S2", [64, 8, 2], "minimal head, same 64-d features (would need training)"),
    ("S3", [48, 24, 12, 2], "kan-image-iot code default (feature_dim 48: CNN projection changes too)"),
]


def head_flash(width: list[int], L: int = LUT_CFG["L"]) -> dict:
    """Bytes of the lut-kan v2 artifact (spline_component, uint8, FP16 metadata, float32 coefficients)."""
    per_edge = {
        "q_table": K_SEGMENTS * L * U8,
        "scale_y_min": 2 * K_SEGMENTS * F16,
        "coef_base_lut_out": 3 * F32,
    }
    edges = sum(a * b for a, b in zip(width[:-1], width[1:]))
    layers = len(width) - 1
    knots = layers * (K_SEGMENTS + 1) * F32
    total = edges * sum(per_edge.values()) + knots
    return {"edges": edges, "bytes_per_edge": per_edge, "knots_bytes": knots, "total_bytes": total}


def head_ram(width: list[int]) -> dict:
    """Two ping-pong float32 buffers sized for the widest layer; LUT tables stay in flash (.rodata)."""
    buf = 2 * max(width) * F32
    return {"activation_buffers_bytes": buf,
            "if_tables_copied_to_sram_bytes": buf + head_flash(width)["total_bytes"]}


IMG, CIN = 128, 3
CONVS = [(3, 16), (16, 24), (24, 32)]
FEAT = 64


def cnn_params() -> dict:
    """Deployment parameters with BatchNorm folded into the preceding conv/linear (weights + bias)."""
    w = sum(ci * co * 9 for ci, co in CONVS) + CONVS[-1][1] * 4 * 4 * FEAT
    b = sum(co for _, co in CONVS) + FEAT
    return {
        "weights": w, "biases": b,
        "float32_bytes": (w + b) * F32,
        "int8_bytes": w * U8 + b * I32 + b * F32,
    }


def cnn_activations() -> list[dict]:
    """Element counts of every tensor along the CNN (batch 1)."""
    t, hw, c = [], IMG, CIN
    t.append(("input", c * hw * hw))
    for i, (ci, co) in enumerate(CONVS, 1):
        t.append((f"conv{i}", co * hw * hw))
        hw //= 2
        t.append((f"pool{i}", co * hw * hw))
    t.append(("avgpool4x4", CONVS[-1][1] * 16))
    t.append(("features", FEAT))
    return [{"tensor": n, "elements": e} for n, e in t]


def cnn_peak(acts: list[dict]) -> dict:
    """Peak live elements for three execution schemes."""
    e = {a["tensor"]: a["elements"] for a in acts}
    order = [a["tensor"] for a in acts]
    naive = max(e[x] + e[y] for x, y in zip(order[:-1], order[1:]))
    blocks, hw, prev = [], IMG, "input"
    for i, (ci, co) in enumerate(CONVS, 1):
        blocks.append(e[prev] + e[f"pool{i}"] + 2 * co * hw)
        prev, hw = f"pool{i}", hw // 2
    fused = max(blocks)
    streamed = max([e["pool1"] + 4 * CIN * IMG + 2 * CONVS[0][1] * IMG] + blocks[1:])
    out = {}
    for name, n in (("op_by_op", naive), ("fused_blocks", fused), ("fused_blocks_streamed_input", streamed)):
        out[name] = {"elements": n, "float32_bytes": n * F32, "int8_bytes": n * U8}
    return out


def cnn_macs() -> int:
    hw, macs = IMG, 0
    for ci, co in CONVS:
        macs += hw * hw * co * ci * 9
        hw //= 2
    return macs + CONVS[-1][1] * 16 * FEAT


def head_budget(width: list[int], n_vectors: int = 19) -> list[dict]:
    """Head-only deployment budget, one row per kind of memory (supervisor's point 4, 16/11)."""
    fl = head_flash(width)
    act_vals = sum(width[1:])                       # h1 + h2 + h3 + logits per vector
    vec_bytes = n_vectors * (width[0] + act_vals) * F32
    return [
        {"item": "Persistent arrays: q tables, FP16 scale/y_min, 3 float32 coefficients per edge, knots",
         "memory": "flash (.rodata, read through the cache)", "bytes": fl["total_bytes"], "status": "computed"},
        {"item": "Activation buffers: two ping-pong float32 buffers of the widest layer",
         "memory": "SRAM", "bytes": head_ram(width)["activation_buffers_bytes"], "status": "computed"},
        {"item": "Workspace: per-input scalars (SiLU, segment, u, w); 4 B per input if SiLU is precomputed per layer",
         "memory": "SRAM (stack)", "bytes": f"0 to {max(width) * F32}", "status": "estimated"},
        {"item": f"Reference vectors for the on-device test ({n_vectors} x (input + h1 + h2 + h3 + logits), float32)",
         "memory": "flash", "bytes": vec_bytes, "status": "estimated (test build only)"},
        {"item": "Stack of the inference task", "memory": "SRAM", "bytes": None,
         "status": "to measure (uxTaskGetStackHighWaterMark)"},
        {"item": "Firmware: code, Arduino-ESP32/ESP-IDF runtime, heap", "memory": "flash + SRAM", "bytes": None,
         "status": "to measure (pio run -t size, ESP.getFreeHeap())"},
    ]


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main() -> None:
    manifest = json.loads((RESULTS_DIR / "lut_manifest.json").read_text())
    built = sum(l["bytes_all_arrays"] for l in manifest["layers"])
    formula = head_flash(manifest["fixture"]["width"])["total_bytes"]
    assert built == formula, f"formula {formula} != built artifact {built}"

    heads = [{"id": i, "width": w, "note": n, "flash": head_flash(w), "ram": head_ram(w)} for i, w, n in HEADS]
    budget = head_budget(HEADS[0][1])
    # cross-check with the trained checkpoint, when checkpoint_head.py has been run (M-hist = H0 shape)
    ck_file = RESULTS_DIR / "checkpoint" / "checkpoint_summary.json"
    ck_check = "checkpoint_summary.json not present"
    if ck_file.exists():
        ck = json.loads(ck_file.read_text())
        ck_check = {"lut_bytes_checkpoint": ck["lut_total_bytes"],
                    "lut_bytes_formula": head_flash(HEADS[0][1])["total_bytes"],
                    "cnn_weights_checkpoint": ck["params"]["cnn_conv_linear_weight_elements"],
                    "cnn_weights_formula": cnn_params()["weights"]}
        assert ck_check["lut_bytes_checkpoint"] == ck_check["lut_bytes_formula"], ck_check
        assert ck_check["cnn_weights_checkpoint"] == ck_check["cnn_weights_formula"], ck_check
    acts = cnn_activations()
    cnn = {"params": cnn_params(), "activations": acts, "peak": cnn_peak(acts), "macs": cnn_macs()}
    h0 = heads[0]
    pipeline = {
        "flash_float32_cnn_plus_head": cnn["params"]["float32_bytes"] + h0["flash"]["total_bytes"],
        "flash_int8_cnn_plus_head": cnn["params"]["int8_bytes"] + h0["flash"]["total_bytes"],
        "test_image_uint8_bytes": CIN * IMG * IMG,
    }
    write_json(RESULTS_DIR / "memory_estimate.json", {
        "assumptions": {
            "lut": {k: LUT_CFG[k] for k in ("L", "dtype", "meta_dtype", "value_representation")},
            "K_segments": K_SEGMENTS,
            "excluded": "firmware baseline, alignment/padding, allocator overhead, stack",
            "head_formula_checked_against_built_artifact_bytes": built,
        },
        "head_budget_H0": budget, "checkpoint_cross_check": ck_check,
        "heads": heads, "cnn": cnn, "pipeline_H0": pipeline,
        "cnn_status": "analytical scenario: int8 quantisation and block fusion are not implemented or measured",
    })

    md = ["| Item | Memory | Bytes | Status |", "|---|---|---|---|"]
    for b in budget:
        md.append(f"| {b['item']} | {b['memory']} | {fmt(b['bytes']) if isinstance(b['bytes'], int) else (b['bytes'] or '-')} | {b['status']} |")
    md += ["", "| Id | Head | Edges | Flash (B) | RAM, tables in flash (B) | RAM, tables copied to SRAM (B) |",
           "|---|---|---|---|---|---|"]
    for h in heads:
        md.append(f"| {h['id']} | {'→'.join(map(str, h['width']))} | {fmt(h['flash']['edges'])} | "
                  f"{fmt(h['flash']['total_bytes'])} | {fmt(h['ram']['activation_buffers_bytes'])} | "
                  f"{fmt(h['ram']['if_tables_copied_to_sram_bytes'])} |")
    md += ["", "| Scheme | Peak live elements | float32 (B) | int8 (B) |", "|---|---|---|---|"]
    for k, v in cnn["peak"].items():
        md.append(f"| {k} | {fmt(v['elements'])} | {fmt(v['float32_bytes'])} | {fmt(v['int8_bytes'])} |")
    md += ["", f"CNN parameters (BN folded): {fmt(cnn['params']['weights'])} weights + {fmt(cnn['params']['biases'])} biases; "
           f"float32 {fmt(cnn['params']['float32_bytes'])} B, int8 {fmt(cnn['params']['int8_bytes'])} B; "
           f"{fmt(cnn['macs'])} MAC per image.",
           f"Pipeline flash (CNN + H0): float32 {fmt(pipeline['flash_float32_cnn_plus_head'])} B, "
           f"int8 CNN {fmt(pipeline['flash_int8_cnn_plus_head'])} B; one 128x128 RGB test image {fmt(pipeline['test_image_uint8_bytes'])} B."]
    (RESULTS_DIR / "memory_estimate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
