# C3-02 – Model contract and porting perimeter

First checkpoint · Nov 15, 2026 · @Fred

First feedback on C3-02: sources, candidate model, LUT export format, I/O schema with reference vectors, and Flash/RAM estimates. No training and no porting have been done.

Main finding: **no trained checkpoint is available in the shared repositories.** The reference vectors therefore use a fixed, untrained fixture head (Section 2). All numbers come from the scripts in `c3-02/` of [esp32c3-kan-person-detection](https://github.com/RodaRoller/esp32c3-kan-person-detection), tag `c3-02-r1`.

## 1. Sources and commits

| Source | Commit | Used for |
| --- | --- | --- |
| [kan-image-iot](https://github.com/KuznetsovKarazin/kan-image-iot) | `c2e17e9` | Model definition, preprocessing, historical run config and metrics |
| [lut-kan](https://github.com/KuznetsovKarazin/lut-kan) | `0041397` (v2.2.0) | LUT v2 builder, NPZ format, reference forward |
| [lut-kan-ids](https://github.com/KuznetsovKarazin/lut-kan-ids) | `b0660d0` | Existing ESP32-C3 engine; not used for the vectors |
| [pykan](https://github.com/KindXiaoming/pykan) | `ecde4ec` (0.2.8) | Float model; commit pinned by lut-kan |
| [esp32c3-kan-person-detection](https://github.com/RodaRoller/esp32c3-kan-person-detection) | tag `c3-02-r1` | This analysis (`c3-02/`) |

Python 3.13.18, torch 2.9.1 (CPU), numpy 2.5.4; full pins in `c3-02/requirements.txt`.

## 2. Candidate model and checkpoint

No trained weights, for either the CNN or the head, are present in the three repositories.

| Item | What is there | Usable for the port |
| --- | --- | --- |
| Historical run `kan_64_24-16-8_grid5_deg3_img128_…`, head 64→24→16→8→2 | `config.json`, metrics (test accuracy 82.0 %, ROC AUC 0.899), parameter counts | Architecture yes; weights not found |
| `kan-image-iot/model/0.0_state`, width 64→32→24→16→2 | pykan's initial state: `history.txt` = "init => 0.0", `scale_sp` = 1/√in\_dim on every edge | No: untrained, different width |
| CNN weights | none | — |

The recorded parameter counts of the historical run (78,544 total, CNN 43,976, KAN 34,568) are reproduced exactly from the code, with 3×3 kernels in all three convolutions.

Proposal, with the two models always labelled separately:

- **F0 — fixture (now):** pykan initial weights for 64→24→16→8→2, frozen in `fixture/fixture_head_64-24-16-8-2_seed0.pt` (SHA-256 `81a67e21…a2ca`). It tests interface and numerics, not detection.
- **M0 — trained model (to be agreed):** the CNN + head checkpoint of the historical run, if it exists in the group archive; otherwise a smaller head from Section 5, which would need training later.

## 3. LUT export format

Each KAN layer is exported as one lut-kan `LUTArtifact` v2 file (`save_lut_npz`): 4 files, 479,040 bytes of arrays for the F0 head (`results/lut/`, manifest in `results/lut_manifest.json`).

| Field | Value |
| --- | --- |
| `knots` | float32\[12\], shared by a layer: −2.2 … 2.2 (grid 5, k 3) → 11 segments |
| `L` | 16 samples per segment, both knots included (`endpoint_inclusive`) |
| `q_table` | uint8 \[E, 11, 16\]; y = y\_min + scale · q, range per segment |
| `scale`, `y_min` | float16 \[E, 11\] |
| `value_representation` | `spline_component`: SiLU base branch computed at runtime |
| `edge_base_scale`, `edge_spline_scale`, `edge_out_scale` | float32 \[E\] = pykan `scale_base`, `scale_sp`, `mask` |
| `oob_behavior`, `boundary_mode` | `zero` (spline only), `closed` |

| Layer | in → out | Edges | Bytes |
| --- | --- | --- | --- |
| 0 | 64 → 24 | 1,536 | 356,400 |
| 1 | 24 → 16 | 384 | 89,136 |
| 2 | 16 → 8 | 128 | 29,744 |
| 3 | 8 → 2 | 16 | 3,760 |
| Total |  | 2,064 | 479,040 |

## 4. Input/output schema and reference vectors

The head takes a float32\[64\] feature vector and returns two float32 logits; the decision is the argmax.

| Stage | Shape | Content |
| --- | --- | --- |
| Image input | float32 \[3, 128, 128\] | RGB resized to 128×128, scaled to \[0, 1\], normalized with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) |
| CNN | — | 3 × \[conv 3×3 → BN → ReLU → max-pool 2\] (16, 24, 32 channels) → avg-pool 4×4 → linear 512→64 → BN → ReLU |
| Head input x | float32 \[64\] | ≥ 0 (after ReLU); LUT domain \[−2.2, 2.2\] |
| Hidden | h1 \[24\], h2 \[16\], h3 \[8\] | plain sums of edge functions |
| Output | logits float32 \[2\] | order \[no\_person, person\] |
| Decision | — | person if z1 > z0, else no\_person |

`results/reference_vectors.json` holds 14 input vectors with the expected h1, h2, h3 and logits from two references: pykan (float model) and lut-kan `forward_reference` (LUT model, the target for the C++ engine). The vectors cover zeros, every knot including ±2.2, exact LUT sample points and midpoints, inputs outside the domain (up to ±10), non-negative inputs like the real features, random inputs, and two inputs giving the person class.

| Vector | pykan logits | LUT reference logits | Decision |
| --- | --- | --- | --- |
| V01\_knots | 0.0209163, −0.00358311 | 0.0208980, −0.00356400 | no\_person |
| V05\_out\_of\_domain | 0.0644920, −0.0549754 | 0.0645589, −0.0550075 | no\_person |
| V12\_person\_branch\_1 | −0.0515223, 0.0208784 | −0.0515206, 0.0209030 | person |

Over the 14 vectors the LUT reference differs from pykan by at most 6.7e-5 on the logits (largest logit 0.065), and the decision is the same in 14 of 14. Because F0 is untrained, these vectors check the arithmetic only, not detection accuracy.

## 5. Flash and RAM estimates

The head fits: 479,040 bytes of tables in flash and 512 bytes of RAM. The full pipeline fits only with an int8 CNN and fused conv/pool blocks, at a peak of about 119 kB of activations.

Budget: 400 KB of SRAM (16 KB of it used as cache); app partition 1.25 MB in the default Arduino-ESP32 layout for 4 MB flash. Firmware baseline and free heap are not included; they will be measured on the build.

### Head only

| Id | Head | Edges | Flash (B) | RAM (B) |
| --- | --- | --- | --- | --- |
| H0 | 64→24→16→8→2 (historical; same shape as F0) | 2,064 | 479,040 | 512 |
| S1 | 64→16→8→2 | 1,168 | 271,120 | 512 |
| S2 | 64→8→2 | 528 | 122,592 | 512 |
| S3 | 48→24→12→2 (kan-image-iot code default) | 1,464 | 339,792 | 384 |

Each edge costs 232 bytes and each layer adds 48 bytes of knots; the formula matches the built F0 artifact. RAM is two float32 buffers sized for the widest layer, with the tables kept in flash. Copying the tables into SRAM, as lut-kan-ids does, is not possible for H0. S1 and S2 keep the 64-d features, so the CNN stays the same, but they would need training.

### Full pipeline (historical CNN + H0, 128×128 input)

| Execution scheme | Peak live elements | float32 (B) | int8 (B) |
| --- | --- | --- | --- |
| Op by op, full tensors | 327,680 | 1,310,720 | 327,680 |
| Conv + BN + ReLU + pool fused per block | 118,784 | 475,136 | 118,784 |
| Fused blocks, input streamed by rows | 93,184 | 372,736 | 93,184 |

CNN weights with BN folded: 174,816 bytes in float32, 44,656 bytes in int8. Total flash with H0: 653,856 bytes (float32 CNN) or 523,696 bytes (int8 CNN). The CNN needs 28.3 million MAC per image, all emulated in software in float32 (no FPU). A float32 pipeline does not fit in SRAM with any margin; an int8 CNN needs the trained checkpoint and calibration images.

## 6. Open questions

1. **Checkpoint.** Does a trained CNN + head checkpoint of the historical run exist (for example in `KAN_Thesis_Code_1210.rar`)? How can I access it, and which file and hash is the reference?
2. **Model.** For M0: the historical H0, or a smaller head (S1, S2), knowing that a smaller one needs training later?
3. **Perimeter.** I propose the KAN/LUT head on the ESP32-C3 as the main result and the full pipeline as a documented feasibility analysis. Do you agree?
4. **pykan version** used for training: `requirements.txt` says only `pykan>=0.1.0`.
5. **Target.** Which board for the hardware test (`esp32-c3-devkitm-1`, as in lut-kan-ids?) and which toolchain version should I pin?

## Reproducibility and authorship

The results are regenerated by three scripts (`inventory.py`, `build_reference.py`, `estimate_memory.py`); setup and commands are in `c3-02/README.md`, and two consecutive runs give byte-identical outputs. Prepared with AI assistance (Claude); every number in this document is read from `c3-02/results/`.
