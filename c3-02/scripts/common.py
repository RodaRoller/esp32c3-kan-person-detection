"""Shared constants and helpers for the C3-02 scripts.

Everything that defines the model contract lives here, so that the three
scripts (inventory, build_reference, estimate_memory) cannot drift apart.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

# Folder layout: c3-02/scripts/common.py -> c3-02/
C302_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = C302_DIR / "results"
FIXTURE_DIR = C302_DIR / "fixture"

# Upstream commits this analysis was produced against (git rev-parse --short HEAD).
PINNED_COMMITS = {
    "kan-image-iot": "c2e17e9",
    "lut-kan": "0041397",
    "lut-kan-ids": "b0660d0",
}

# pykan commit pinned in lut-kan/requirements-lock-win.txt (reports as version 0.2.8).
PYKAN_COMMIT = "ecde4ec3274d3bef1ad737479cf126aed38ab530"

# Historical reference run in kan-image-iot/experiment_data.
HISTORICAL_RUN = "kan_64_24-16-8_grid5_deg3_img128_bs64_lr0.002_wd1e-05_do0.05"

# Thesis LUT contract (lut-kan format_version 2). Argument names follow
# lut-kan/src/quant/lut_builder.py::build_lut_for_edges.
LUT_CFG = {
    "L": 16,                                  # samples per segment, endpoints included
    "interp": "linear",
    "y_range_method": "minmax",
    "lower_pct": 0.0,                         # unused with minmax
    "upper_pct": 100.0,                       # unused with minmax
    "dtype": "uint8",
    "scheme": "asymmetric",                   # y = y_min + scale * q
    "qmin": 0,
    "qmax": 255,
    "meta_dtype": "float16",                  # scale and y_min per segment
    "value_representation": "spline_component",  # LUT holds the spline only
    "oob_behavior": "zero",                   # spline = 0 outside the domain, base branch kept
    "boundary_mode": "closed",                # domain is [x_min, x_max]
}

# Deterministic head used for the contract vectors. It is NOT a trained model:
# pykan initial weights for the historical architecture, fixed by the seed.
FIXTURE = {"width": [64, 24, 16, 8, 2], "grid": 5, "k": 3, "seed": 0}
FIXTURE_NAME = "fixture_head_64-24-16-8-2_seed0"

CLASS_NAMES = ["no_person", "person"]  # order of the two logits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_short_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def check_pinned(name: str, repo: Path) -> str:
    """Return the repo HEAD and warn loudly if it is not the pinned commit."""
    head = git_short_head(repo)
    if head != PINNED_COMMITS[name]:
        print(f"WARNING: {name} is at {head}, expected {PINNED_COMMITS[name]}", file=sys.stderr)
    return head


def environment() -> dict:
    import numpy
    env = {"python": platform.python_version(), "numpy": numpy.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    env["pykan_commit"] = PYKAN_COMMIT
    return env


def f32(v) -> float:
    """float32 -> JSON number with 9 significant digits (round-trips to the same float32)."""
    return float(f"{float(v):.9g}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
