# C3-02 — Model contract and porting perimeter

Material for the first C3-02 checkpoint: which sources and model the porting starts from, the LUT export format, the input/output schema with reference vectors, and Flash/RAM estimates for the KAN head alone and for the full CNN + head pipeline.

No model is trained here and nothing runs on the ESP32-C3 yet.

## Important: the fixture is not a trained model

No trained checkpoint (CNN or head) is available in the public repositories. The vectors are therefore computed on a **fixture**: pykan's initial weights for the historical head shape 64→24→16→8→2, frozen in `fixture/fixture_head_64-24-16-8-2_seed0.pt`. The fixture tests the interface and the numerics of the LUT engine. It does not test person detection. When the trained checkpoint is available, the same scripts will be re-run on it.

## Contents

| Path | What it is |
|---|---|
| `scripts/common.py` | Pinned commits, LUT contract parameters, fixture definition |
| `scripts/inventory.py` | Checks commits, lists weight files, inspects `model/0.0_state`, rebuilds the historical parameter counts |
| `scripts/build_reference.py` | Fixture → lut-kan LUT v2 export (one `.npz` per layer) → reference vectors |
| `scripts/estimate_memory.py` | Analytic Flash/RAM estimates (head configurations and CNN pipeline) |
| `fixture/` | Fixture head (pykan `state_dict`) |
| `results/inventory.json` | Output of `inventory.py` |
| `results/lut/layer{0..3}.npz`, `results/lut_manifest.json` | LUT export and its manifest (parameters, shapes, sizes, hashes) |
| `results/reference_vectors.json` | 14 input vectors with expected pykan and LUT-reference outputs per layer |
| `results/memory_estimate.{json,md}` | Memory estimates |

## Reproduce

Clone the upstream sources next to this repository, not inside it:

```bash
mkdir upstream && cd upstream
git clone https://github.com/KuznetsovKarazin/kan-image-iot && git -C kan-image-iot checkout c2e17e9
git clone https://github.com/KuznetsovKarazin/lut-kan       && git -C lut-kan checkout 0041397
git clone https://github.com/KuznetsovKarazin/lut-kan-ids   && git -C lut-kan-ids checkout b0660d0
cd ..
```

Then, from `esp32c3-kan-person-detection/c3-02`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
python scripts/inventory.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan --lut-kan-ids ../../upstream/lut-kan-ids
python scripts/build_reference.py --lut-kan ../../upstream/lut-kan
python scripts/estimate_memory.py
```

`.venv/` must not be committed (add it to `.gitignore`).

`build_reference.py` loads the committed fixture file. Re-running it gives byte-identical `reference_vectors.json` and LUT content. `--regenerate-fixture` creates new pykan initial weights; pykan's initialisation is not bit-reproducible, so every expected value would change.

## Own work vs reused

- Reused: pykan (model and B-splines), lut-kan (`build_lut_for_edges`, `save_lut_npz`, `pack_dense_layer`, `forward_reference`, `PyKANSingleLayerAdapter`), kan-image-iot (`KANImageClassifier` for the parameter counts). None of their files is modified.
- Own: the scripts in `scripts/`, the choice of vectors and the memory model.
- Prepared with AI assistance (Claude). The numbers in the C3-02 document are taken from the files in `results/`.
