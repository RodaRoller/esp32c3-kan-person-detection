# C3-02 — Model contract and porting perimeter

Which model is ported, in which LUT format, with which reference vectors and acceptance criterion, and how much Flash/RAM it needs. The report is [`C3-02_model_contract.md`](C3-02_model_contract.md) (r2, final). The first draft is at tag `c3-02-r1`. Approved by the supervisor on Nov 24, 2026.

No model is trained here and nothing runs on the ESP32-C3 yet.

## Two models, kept apart

| Id | What | Where |
|---|---|---|
| F0 | Untrained fixture: pykan initial weights, head 64→24→16→8→2 | Public: `fixture/` |
| M-hist | Historical trained checkpoint (CNN + head 64→24→16→8→2) of a previous thesis in the group | **Private**: not in this repository |

## The private checkpoint

The trained weights, and everything derived from them (LUTs, reference vectors, per-image scores), are not published, as agreed with the supervisor. To run the M-hist steps, place the file here, relative to the repository root:

```
private/checkpoint/kan_person_detector_best.pt
SHA-256 03b4132217626a7a4e3fc91813ad84ea9794152409e042569ec515c20c5d9ffc
```

Source: package `KAN_Thesis_Code_FINAL.rar` (Oct 6, 2026), file `artifacts/checkpoint/kan_person_detector_best.pt`, available from the supervisor. The scripts check the hash and stop if it differs. `private/` is listed in `.gitignore`; the M-hist scripts write their outputs to `private/c3-02/`.

Optional, for the comparisons with the historical results: the package folders `artifacts/lut_L16/` and `results/`, passed with `--archive-lut` and `--archive-results`.

## Contents

| Path | What it is |
|---|---|
| `C3-02_model_contract.md` | The C3-02 report |
| `scripts/common.py` | Pinned commits, LUT contract parameters, F0 and M-hist definitions |
| `scripts/inventory.py` | (r1) Checks commits, lists weight files in the public repositories, rebuilds the historical parameter counts |
| `scripts/build_reference.py` | (r1) F0 → LUT v2 export → 14 public reference vectors |
| `scripts/estimate_memory.py` | Flash/RAM: per-kind budget of the head, head configurations, CNN pipeline scenarios |
| `scripts/checkpoint_head.py` | M-hist: hash, safe and strict load, structure checks, LUT v2 export, comparison with the historical LUTs |
| `scripts/evaluate_checkpoint.py` | M-hist on the validation and test images: pykan vs LUT reference, reproduction of the archived scores, domain and margin statistics, split overlap |
| `scripts/tolerance_and_vectors.py` | float32 model of the future C++ loop, legitimate and faulty variants → tolerance; 19 M-hist reference vectors |
| `fixture/` | F0 head (pykan `state_dict`) |
| `results/` | F0 outputs (r1): LUT export, manifest, reference vectors; memory estimates |
| `results/checkpoint/` | M-hist outputs that can be public: metadata, hashes, aggregate metrics, index of the private vectors |

## Reproduce

Clone the upstream sources next to this repository, not inside it:

```bash
mkdir upstream && cd upstream
git clone https://github.com/KuznetsovKarazin/kan-image-iot && git -C kan-image-iot checkout c2e17e9
git clone https://github.com/KuznetsovKarazin/lut-kan       && git -C lut-kan checkout 0041397
git clone https://github.com/KuznetsovKarazin/lut-kan-ids   && git -C lut-kan-ids checkout b0660d0
cd ..
```

`evaluate_checkpoint.py` needs the images in `kan-image-iot/data/processed/vww_subset/{train,val,test}` (prepared by kan-image-iot; pass another location with `--data`).

Then, from `esp32c3-kan-person-detection/c3-02`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt

# F0 (public, r1)
python scripts/inventory.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan --lut-kan-ids ../../upstream/lut-kan-ids
python scripts/build_reference.py --lut-kan ../../upstream/lut-kan

# M-hist (needs private/checkpoint/, see above)
python scripts/checkpoint_head.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan --archive-lut <package>/artifacts/lut_L16
python scripts/evaluate_checkpoint.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan --archive-results <package>/results
python scripts/tolerance_and_vectors.py --kan-image-iot ../../upstream/kan-image-iot --lut-kan ../../upstream/lut-kan

# Memory (reads the outputs above)
python scripts/estimate_memory.py
```

The steps are deterministic: re-running them gives the same public results. `evaluate_checkpoint.py` runs single-threaded (about one minute on a laptop CPU) so that the float32 sums do not depend on the thread count. Building the model makes pykan create a `model/` folder in the working directory; it is git-ignored.

## Own work vs reused

- Reused, unmodified:
  - pykan;
  - lut-kan (`build_lut_for_edges`, `save_lut_npz`, `load_lut_npz`, `pack_dense_layer`, `forward_reference`, `PyKANSingleLayerAdapter`);
  - kan-image-iot (`KANImageClassifier`, evaluation preprocessing, dataset);
  - the historical package (checkpoint, LUT artifacts and scores, as references only).
- Own: the scripts in `scripts/`, the choice of vectors, the tolerance measurement and the report.
- Prepared with AI assistance (Claude). Reviewd and checked by me.
