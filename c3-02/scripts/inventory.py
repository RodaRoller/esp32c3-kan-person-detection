"""C3-02 step 1: inventory of sources, commits and candidate checkpoints.

Usage:
    python scripts/inventory.py --kan-image-iot ../ext/kan-image-iot \
        --lut-kan ../ext/lut-kan --lut-kan-ids ../ext/lut-kan-ids

Writes results/inventory.json. Does not modify the upstream repositories.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import HISTORICAL_RUN, RESULTS_DIR, check_pinned, environment, sha256, write_json

WEIGHT_EXT = (".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy", ".onnx", ".h5",
              ".pkl", ".bin", ".tflite", ".rar", ".zip", "_state")


@contextlib.contextmanager
def scratch_cwd():
    """pykan's KAN() saves ./model/0.0_* on construction (auto_save=True).
    Build models inside a throw-away folder so nothing is written into the repos."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(old)


def tracked_weight_files(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True)
    return [p for p in out.stdout.splitlines() if p.lower().endswith(WEIGHT_EXT)]


def inspect_pykan_ckpt(repo: Path) -> dict:
    """model/0.0_state: is it a trained head or pykan's initial state?"""
    import torch
    import yaml

    folder = repo / "model"
    cfg = yaml.safe_load((folder / "0.0_config.yml").read_text())
    sd = torch.load(folder / "0.0_state", map_location="cpu", weights_only=True)
    width = [w[0] for w in cfg["width"]]

    # pykan initialises scale_sp to exactly 1/sqrt(in_dim) on every edge.
    # Training with sp_trainable=True moves it, so equality on all edges means "not trained".
    layers = []
    for l in range(len(width) - 1):
        sp = sd[f"act_fun.{l}.scale_sp"].numpy()
        expected = 1.0 / np.sqrt(sp.shape[0])
        # one single value on every edge, equal to 1/sqrt(in_dim) up to float32 rounding
        is_init = bool(np.all(sp == sp.flat[0]) and abs(float(sp.flat[0]) - expected) <= 1e-6 * expected)
        layers.append({
            "layer": l,
            "in": int(sp.shape[0]), "out": int(sp.shape[1]),
            "scale_sp_value": float(sp.flat[0]),
            "scale_sp_all_equal_init_value": is_init,
            "coef_abs_max": float(np.abs(sd[f"act_fun.{l}.coef"].numpy()).max()),
        })
    return {
        "path": "model/0.0_state",
        "sha256": sha256(folder / "0.0_state"),
        "width": width,
        "grid": cfg["grid"], "k": cfg["k"], "grid_range": cfg["grid_range"],
        "history_txt": (folder / "history.txt").read_text().strip(),
        "layers": layers,
        "is_initial_state": all(x["scale_sp_all_equal_init_value"] for x in layers),
    }


def historical_run(repo: Path) -> dict:
    run = repo / "experiment_data" / HISTORICAL_RUN
    cfg = json.loads((run / "config.json").read_text())
    pre, kan = cfg["preprocessor_config"], cfg["kan_config"]

    sys.path.insert(0, str(repo))
    from src.models.kan_model import KANImageClassifier  # noqa: E402

    # Same arguments as src/train.py (lines ~1013-1027). kernel_size and
    # final_pool_size are NOT passed, so the ImagePreprocessor defaults apply.
    with scratch_cwd():
        model = KANImageClassifier(
            input_channels=pre["input_channels"], conv_channels=pre["conv_channels"],
            img_size=cfg["dataset_config"]["img_size"], num_classes=2,
            feature_dim=kan["feature_dim"], kan_hidden_dims=kan["hidden_dims"],
            kan_grid=kan["grid"], kan_degree=kan["degree"],
            use_batch_norm=pre["use_batch_norm"], dropout_rate=kan["dropout_rate"],
            activation_l1=kan["activation_l1"],
            stochastic_depth_rate=pre.get("stochastic_depth_rate", 0.0), seed=kan["seed"],
        )
    counts = model.get_parameter_count()
    convs = [(n, list(p.shape)) for n, p in model.preprocessor.named_parameters()
             if n.endswith("0.weight") and p.ndim == 4]

    recorded = {}
    for line in (run / "analysis" / "model_architecture.txt").read_text().splitlines():
        for key in ("Total parameters", "Trainable parameters", "CNN parameters", "KAN parameters"):
            if line.startswith(key + ":"):
                recorded[key] = int(line.split(":")[1].split("(")[0].strip().replace(",", ""))

    return {
        "run": HISTORICAL_RUN,
        "kan_width": [kan["feature_dim"], *kan["hidden_dims"], 2],
        "img_size": cfg["dataset_config"]["img_size"],
        "config_kernel_size": pre["kernel_size"],
        "instantiated_conv_weights": convs,
        "param_counts_rebuilt": counts,
        "param_counts_recorded": recorded,
        "counts_match": (counts["total"], counts["trainable"], counts["cnn"], counts["kan"]) == (
            recorded["Total parameters"], recorded["Trainable parameters"],
            recorded["CNN parameters"], recorded["KAN parameters"]),
        "weights_in_repo": False,  # no tracked file of this run holds weights (see tracked_weight_files)
        "run_files": sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kan-image-iot", type=Path, required=True)
    ap.add_argument("--lut-kan", type=Path, required=True)
    ap.add_argument("--lut-kan-ids", type=Path, required=True)
    a = ap.parse_args()

    repos = {"kan-image-iot": a.kan_image_iot, "lut-kan": a.lut_kan, "lut-kan-ids": a.lut_kan_ids}
    inv = {
        "environment": environment(),
        "repos": {n: {"head": check_pinned(n, p), "tracked_weight_files": tracked_weight_files(p)}
                  for n, p in repos.items()},
        "kan_image_iot_model_folder": inspect_pykan_ckpt(a.kan_image_iot),
        "historical_run": historical_run(a.kan_image_iot),
    }
    write_json(RESULTS_DIR / "inventory.json", inv)

    k = inv["kan_image_iot_model_folder"]
    h = inv["historical_run"]
    print("tracked weight files:", {n: r["tracked_weight_files"] for n, r in inv["repos"].items()})
    print(f"model/0.0_state width={k['width']} initial_state={k['is_initial_state']} history={k['history_txt']!r}")
    print(f"historical run: params rebuilt={h['param_counts_rebuilt']} match_recorded={h['counts_match']}")
    print(f"  config kernel_size={h['config_kernel_size']} instantiated={h['instantiated_conv_weights']}")


if __name__ == "__main__":
    main()
