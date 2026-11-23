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



C302_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = C302_DIR / "results"
FIXTURE_DIR = C302_DIR / "fixture"

PINNED_COMMITS = {
    "kan-image-iot": "c2e17e9",
    "lut-kan": "0041397",
    "lut-kan-ids": "b0660d0",
}

PYKAN_COMMIT = "ecde4ec3274d3bef1ad737479cf126aed38ab530"

# Historical reference run in kan-image-iot/experiment_data.
HISTORICAL_RUN = "kan_64_24-16-8_grid5_deg3_img128_bs64_lr0.002_wd1e-05_do0.05"

LUT_CFG = {
    "L": 16,                                  
    "interp": "linear",
    "y_range_method": "minmax",
    "lower_pct": 0.0,                         
    "upper_pct": 100.0,                       
    "dtype": "uint8",
    "scheme": "asymmetric",                   
    "qmin": 0,
    "qmax": 255,
    "meta_dtype": "float16",                  
    "value_representation": "spline_component",  
    "oob_behavior": "zero",                   
    "boundary_mode": "closed",                
}


FIXTURE = {"width": [64, 24, 16, 8, 2], "grid": 5, "k": 3, "seed": 0}
FIXTURE_NAME = "fixture_head_64-24-16-8-2_seed0"

CLASS_NAMES = ["no_person", "person"]  # order of the two logits


REPO_DIR = C302_DIR.parent
PRIVATE_DIR = REPO_DIR / "private"
PRIVATE_C302 = PRIVATE_DIR / "c3-02"
CHECKPOINT = {
    "id": "M-hist",
    "file": "kan_person_detector_best.pt",
    "sha256": "03b4132217626a7a4e3fc91813ad84ea9794152409e042569ec515c20c5d9ffc",
    "local_path": "private/checkpoint/kan_person_detector_best.pt",   # relative to the repo root
    "source": "KAN_Thesis_Code_FINAL.rar (6 Oct 2026), artifacts/checkpoint/",
    "expected": {"epoch": 49, "batch_size": 512, "used_ema": True},
}
EXP01_MODEL_ARGS = {
    "input_channels": 3, "conv_channels": [16, 24, 32], "img_size": 128, "num_classes": 2,
    "feature_dim": 64, "kan_hidden_dims": [24, 16, 8], "kan_grid": 5, "kan_degree": 3,
    "use_batch_norm": True, "dropout_rate": 0.05, "activation_l1": 1e-5,
    "stochastic_depth_rate": 0.1, "seed": 42,
}
EVAL_MEAN = (0.485, 0.456, 0.406)
EVAL_STD = (0.229, 0.224, 0.225)
HISTORICAL_THRESHOLD_P = 0.37709295749664307


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
