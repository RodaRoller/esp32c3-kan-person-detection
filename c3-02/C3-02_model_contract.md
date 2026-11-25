# C3-02 – Model contract and porting perimeter

Final checkpoint (r2) · Nov 23, 2026 · supersedes the first draft (tag `c3-02-r1`, Nov 15) · approved Nov 24, 2026.

The trained historical checkpoint (**M-hist**) loads strictly into the public kan-image-iot model and reproduces the archived scores. Its LUT v2 export is byte-identical to the historical LUT artifacts. The supervisor approved this perimeter on Nov 24, 2026:

- **First functional result:** the KAN/LUT head of M-hist on the ESP32-C3, fed with precomputed features.
- **Kept separate:** the full camera → CNN → head pipeline, assessed only as an analytical scenario.
- **Acceptance criterion:** |C++ − LUT reference| ≤ 2·10⁻⁶·max(1, |reference|) on every layer, with identical decisions.

Changes since r1, following the four points of the review of Nov 16:

1. F0, M-hist and the smaller configurations S1–S3 are identified separately (§2).
2. kan-image-iot is the model reference and lut-kan the export and numeric contract. The IDS engine is compared row by row (§5).
3. Tolerance, boundaries, out-of-domain inputs, NaN/Inf, ties, and the decision rule versus the historical threshold are defined (§6).
4. The memory figures are split into persistent arrays, activation buffers, workspace, stack and firmware. CNN figures are marked as estimates (§7).

The public numbers come from `c3-02/results/`. The trained weights and everything derived from them stay private (§11).

## 1. Sources and commits

| Source | Commit / version | Used for |
| --- | --- | --- |
| [kan-image-iot](https://github.com/KuznetsovKarazin/kan-image-iot) | `c2e17e9` | Model class `KANImageClassifier`, evaluation preprocessing, VWW subset (train 100,000 / val 4,830 / test 4,000) |
| [lut-kan](https://github.com/KuznetsovKarazin/lut-kan) | `0041397` (v2.2.0) | LUT v2 builder, NPZ format, `forward_reference` (the numeric contract) |
| [lut-kan-ids](https://github.com/KuznetsovKarazin/lut-kan-ids) | `b0660d0` | Existing ESP32-C3 engine, compared in §5 |
| [pykan](https://github.com/KindXiaoming/pykan) | `ecde4ec` (0.2.8) | Float model; same version as the historical training |
| Historical package `KAN_Thesis_Code_FINAL.rar` (Oct 6, 2026), from a previous thesis of the group | private | Checkpoint, historical LUT artifacts and scores, used as references only |
| This repository, `c3-02/` | tag `c3-02-r2` | Scripts and public results |

Environment: Python 3.13.18, torch 2.9.1 (CPU), torchvision 0.24.1, numpy 2.5.4. The package was trained with torch 2.11 on a GPU. The reproduction differs from it by at most 2.1·10⁻⁶ on the logits (§2).

## 2. Models: F0, M-hist, S1–S3

| Id | What it is | Trained | Weights | Role |
| --- | --- | --- | --- | --- |
| F0 | pykan initial weights, head 64→24→16→8→2, `fixture/fixture_head_64-24-16-8-2_seed0.pt` (SHA-256 `81a67e21…a2ca`) | no | public | Interface and arithmetic vectors (r1, unchanged) |
| M-hist | `kan_person_detector_best.pt` (SHA-256 `03b41322…9ffc`): CNN + head 64→24→16→8→2, epoch 49, batch 512, EMA weights | yes | private | Model of the port: LUT export, reference vectors, functional checks |
| S1–S3 | smaller heads (§7) | no | none | Not proposed now; each would need training, its own Id and separate results |

**Checks on M-hist** (`scripts/checkpoint_head.py` → `results/checkpoint/checkpoint_summary.json`):

- **Hash.** The SHA-256 is the one recorded on Trello. The file is opened with `torch.load(weights_only=True)`, so no code stored in it is executed.
- **Strict load.** It loads into `KANImageClassifier` with the EXP-01 constructor arguments, and all keys match.
- **Convolutions.** There are three 3×3 convolutions (16, 24, 32 channels): the training code does not pass `kernel_size`, so the `[3, 3, 1]` in the config is not applied.
- **Parameter counts.** 78,544 = CNN 43,976 + KAN 34,568. These are exactly the counts recorded for the historical run.
- **The head is a plain sum.**
  - Node and subnode affines are the identity.
  - The symbolic branch is off and every edge mask is 1.
  - Each layer has a single grid, [−2.2, 2.2].

  So contract v2 represents it with no extra operation.
- **Reproduction** (`scripts/evaluate_checkpoint.py` → `results/checkpoint/test_evaluation.json`):
  - The float logits match the archived scores within 2.1·10⁻⁶, and every decision is identical on validation and test.
  - Validation accuracy is 81.88 %, the value stored in the checkpoint.
  - On the test set: accuracy 81.40 % at argmax, 81.63 % at the historical threshold. These figures were not computed on held-out data (next point).

**Limits of M-hist:**

- **The test set is not held out.** All 4,000 test images are among the 4,830 validation images (same file names). The checkpoint (best validation accuracy) and the historical threshold were both selected on validation. Test figures are therefore not independent, and M-hist is used for functional checks only. No validation or test image is in the training set.
- **Package integrity.** In the package, 27 of 28 files match `SHA256SUMS.txt`; the notebook does not. I have not verified the package against the Oct 12 deposit.

## 3. LUT export format

Each KAN layer is one lut-kan `LUTArtifact` v2 file (`save_lut_npz`). The format is unchanged from r1:

| Field | Value |
| --- | --- |
| `knots` | float32[12] shared by a layer, −2.2 … 2.2 (grid 5, k 3) → 11 segments |
| `L` | 16 samples per segment, both knots included (`endpoint_inclusive`) |
| `q_table` | uint8 [E, 11, 16]; y = y_min + scale · q, range per segment |
| `scale`, `y_min` | float16 [E, 11] |
| `edge_base_scale`, `edge_spline_scale`, `edge_out_scale` | float32 [E] = pykan `scale_base`, `scale_sp`, `mask` |
| `value_representation` | `spline_component`: the SiLU base branch is computed at runtime |
| `oob_behavior`, `boundary_mode` | `zero` (spline only), `closed` (`pack_dense_layer` defaults to `half_open`: `closed` must be passed) |

The M-hist export has 4 files and 479,040 bytes of arrays (layers 356,400 / 89,136 / 29,744 / 3,760). All seven arrays of every layer are byte-identical to `artifacts/lut_L16/` of the historical package. The export is therefore the one the historical results were obtained with. It stays in `private/`.

## 4. Input/output schema and reference vectors

| Stage | Shape | Content |
| --- | --- | --- |
| Image | float32 [3, 128, 128] | RGB resized to 128×128, scaled to [0, 1], ImageNet mean/std normalisation |
| CNN | — | 3 × [conv 3×3 → BN → ReLU → max-pool 2] (16, 24, 32 ch) → avg-pool 4×4 → linear 512→64 → BN → ReLU |
| Head input x | float32 [64] | ≥ 0; 0.64 % of the values exceed 2.2 (test) |
| Hidden | h1 [24], h2 [16], h3 [8] | Plain sums of edge functions; values down to −12.35 (validation) |
| Output | float32 [2] | Logits, order [no_person, person] |
| Decision | — | Margin d = z1 − z0 compared with θ (§6) |

**Vector sets:**

- **F0, public:** 14 vectors in `results/reference_vectors.json`, unchanged from r1, already checked by the supervisor.
- **M-hist, private:** 19 vectors in `private/c3-02/reference_vectors_mhist.json`. Its index and SHA-256 are in `results/checkpoint/reference_vectors_mhist_index.json`. Each vector has the expected h1, h2, h3 and logits of the LUT reference.
  - **10 real feature vectors**, chosen for the cases that matter: the most confident image of each class; the closest images on each side of θ = 0 and of the historical threshold; the most out-of-domain inputs in layers 0, 1 and 3; a false negative.
  - **6 synthetic corner cases**, built with the same rules as the F0 set: zeros, every knot including ±2.2, exact samples, sample midpoints, inputs outside the domain.
  - **3 non-finite inputs:** NaN, +Inf and −Inf.

**How well the tables approximate the trained head** (4,830 validation images, which include the test set):

| Quantity | Value |
| --- | --- |
| Logits, LUT reference vs pykan: mean / max absolute difference | 1.97·10⁻⁴ / 1.52·10⁻³ (largest \|logit\| 2.02) |
| Hidden layers, max absolute difference | h1 1.6·10⁻³, h2 1.4·10⁻³, h3 1.9·10⁻³ |
| Same decision as pykan | 4,830 / 4,830 at θ = 0 and at the historical threshold |
| Smallest \|z1 − z0\| of the LUT reference | 1.2·10⁻⁴ (2 images below 10⁻³, no exact tie) |

The share of values outside [−2.2, 2.2] in each layer's input, on the test set:

| Layer input | Below −2.2 | Above 2.2 | Images with at least one value outside |
| --- | --- | --- | --- |
| 0 (features) | 0 % | 0.64 % | 817 / 4,000 |
| 1 (h1) | 2.29 % | 0.40 % | 728 |
| 2 (h2) | 0.54 % | 0.26 % | 154 |
| 3 (h3) | 0.33 % | 1.68 % | 427 |

The out-of-domain branch (spline = 0, SiLU kept) is therefore part of normal operation, not a rare edge case.

## 5. lut-kan-ids engine vs contract v2

Each row names the line that supports it. "IDS" refers to `lut-kan-ids/hardware/ids_esp32_lut_kan/main.cpp` and `lut_kan_model.h` at `b0660d0`. "v2" refers to `lut-kan/src/kernels/lut_backend_reference.py` (ref) and `lut_math.py` (math) at `0041397`.

| # | Aspect | IDS engine | Contract v2 (M-hist) | Adaptation |
| --- | --- | --- | --- | --- |
| 1 | Architecture | 16→8→1 (main L4, L25–28) | 64→24→16→8→2, 4 layers | Engine generic in the layer sizes, sizes from the generated header |
| 2 | Domain | [−1, 1); x clipped to [−1, 1−10⁻⁷] (main L101, L109–110) | [−2.2, 2.2] closed (math L14–18) | Replace the clipping with the domain mask of row 7 |
| 3 | Segment | Uniform: seg = (int)((x − XMIN)·INV_SW), 4 segments (main L111–113) | Search in the stored float32 knots, 11 segments (math L61–77; `uniform_dx = None`) | Use the stored knots. The uniform formula picks the same segment at all 12 knots of this grid (measured, §6), but it is not the contract |
| 4 | Position | Cell: (int)((frac − seg)·8) (main L114–116) | pos = u·(L−1), L = 16, both segment ends sampled (math L81–92) | Different sampling grid: reading v2 tables "by cell" shifts every value |
| 5 | Interpolation | None, the value of the cell (main L118–119) | Linear between samples r0 and r1 (ref L48) | Add it. Without it the logits move by up to 8·10⁻² and 17 decisions change (measured, §6) |
| 6 | Quantisation metadata | Per edge, float32 scale and offset (main L86–87; header L26, L44) | Per edge and per segment, FP16 y_min and scale (ref L42; math L95–101) | Index the metadata by segment; widen FP16 to float32 in software |
| 7 | Tabulated function | The whole φ | Spline part only. Outside the domain the spline is 0 (ref L50–51); the term is m·(w_b·SiLU(x) + w_s·s) (ref L80–83; SiLU math L33–40) | One SiLU per input (112 `expf` per inference) and 3 float32 coefficients per edge |
| 8 | Between layers | Normalisation (z − shift)·inv_scale, then a 256-entry tanh LUT (main L121–123, L73–81) | Nothing: the sum goes straight to the next layer | Remove normalisation and tanh |
| 9 | Output | 1 logit; decision `logit > 0.0f` (main L188) | 2 logits [no_person, person] | Margin d = z1 − z0 and threshold θ (§6) |
| 10 | Loop order | j outer, i inner (main L105–107) | i outer, j inner (ref L74–83) | Each output still accumulates over i in ascending order, so the float32 sums are the same. i outer computes each SiLU once |
| 11 | Table placement | Copied from flash to SRAM with `memcpy` (main L160–168); about 6.5 KB | 479,040 B: cannot be copied into SRAM | `const` arrays in flash (.rodata, read through the cache); measure the latency cost |
| 12 | Divisions | None at runtime: INV_SW and 1/scale precomputed (main L30–31, L170–172) | The reference divides by (t_{k+1} − t_k) (math L76) | Precomputed reciprocals are allowed (measured difference ≤ 3.5·10⁻⁷, §6) |
| 13 | Board and toolchain | `board = esp32-c3-devkitm-1`, `platform = espressif32` without a version (platformio.ini); the comment and the hardware results say SuperMini (main L3; `results/LUT_KAN_NODIV_K4_L8_hw.json`) | — | Pin the platform version; confirm the board (§9) |

What is reused from the IDS engine is the structure: tables as C arrays, loops without divisions, a serial test harness. The numeric rules of rows 2–9 come from contract v2.

## 6. Numeric acceptance criterion and edge cases

**How the tolerance was measured** (`scripts/tolerance_and_vectors.py` → `results/checkpoint/tolerance.json`). A numpy model of the future C++ loop computes, in float32 and in the C order: i outer, one SiLU per input, FP16 metadata widened to float32. It is a measuring tool, not the engine. Without changes it reproduces the lut-kan reference bit for bit.

I then varied it in two ways:

- **Legitimate variants:** what a correct C++ engine can do differently, such as another `expf` (modelled as ±1 ULP), precomputed reciprocals, or SiLU written as x·sigmoid(x).
- **Contract violations:** cell lookup without interpolation, spline not zeroed outside the domain, half-open domain, FP16 arithmetic.

The inputs were the 4,830 validation feature vectors plus 12 synthetic vectors. Errors are relative to max(1, |reference|), on h1, h2, h3 and logits:

| Variant | Kind | Max error | Vectors over 2·10⁻⁶ | Decisions changed (θ = 0) |
| --- | --- | --- | --- | --- |
| Exact float32 replica | legitimate | 0 | 0 | 0 |
| `expf` ±1 ULP | legitimate | 4.1·10⁻⁷ | 0 | 0 |
| Precomputed reciprocals | legitimate | 3.5·10⁻⁷ | 0 | 0 |
| SiLU as x·sigmoid(x) | legitimate | 4.2·10⁻⁷ | 0 | 0 |
| All three together | legitimate | 4.4·10⁻⁷ | 0 | 0 |
| Uniform segment formula | deviation, same segments here | 5.4·10⁻⁷ | 0 | 0 |
| Half-open domain | violation | 1.3·10⁻⁵ | 2 | 0 |
| Spline not zeroed out of domain | violation | 2.7·10⁻⁵ | 847 | 0 |
| FP16 arithmetic on metadata | violation | 2.6·10⁻⁴ | 4,842 | 0 |
| Cell lookup, no interpolation | violation | 9.9·10⁻² | 4,842 | 17 |

Two observations follow from the table. Some violations change no decision on this data, so checking decisions alone would miss them. And the largest legitimate error (4.4·10⁻⁷) is about 30 times smaller than the smallest violation (1.3·10⁻⁵).

**Rules.**

1. **Tolerance.** |C++ − LUT reference| ≤ 2·10⁻⁶·max(1, |reference|) on every h1, h2, h3 and logit of every vector, and identical decisions.
   - 2·10⁻⁶ is about 4.5 times above the largest legitimate difference and about 6 times below the smallest violation. A correct engine passes; every violation listed fails on at least one vector.
   - It is relative because hidden values reach 12, and absolute near zero.
   - Identical decisions can be required because the smallest margin (1.2·10⁻⁴) is 60 times larger than the tolerance.
   - It applies to the float32 engine; a fixed-point variant would get its own criterion, measured the same way.
   - pykan against the LUT is a different question (§4): reported, not part of the acceptance.
2. **Boundaries.** The domain is closed: x = ±2.2 is inside. A knot belongs to the segment on its right, except x = 2.2, which is the end of the last segment (u = 1). The segment comes from the stored knots, not from x_min + k·dx. Covered by S01, S02 and S03.
3. **Out of domain.** The spline term is 0; the SiLU term is computed on the unclipped x. Covered by S05 and by the real vectors R07–R09.
4. **NaN and ±Inf.** The engine rejects the input: it returns an error status and no decision.
   - In the reference all three cases end in NaN logits (N01–N03), and that is a side effect, not a specification.
   - In C/C++, converting NaN or Inf to the segment index is undefined behaviour.
   - A non-finite feature means a fault upstream.
   - One `isfinite` check per input (64) is enough: with finite inputs every later value is finite.
   - This is a declared deviation from the reference.
5. **Ties.** z1 = z0 gives no_person, so the rule is "person if d > θ", with strict inequality. It is the convention of argmax in torch and numpy, and in kan-image-iot (`analyze.py` L207). The historical package applied p ≥ t, which differs only on exact equality; no exact tie z1 = z0 occurs in the 4,830 validation images.
6. **Decision rule vs historical threshold.** The engine outputs the margin d = z1 − z0 and compares it with a threshold θ declared as a build parameter.
   - **Conformance (θ = 0):** this is the argmax, the rule of kan-image-iot and of the reference. All acceptance tests use it.
   - **Historical operating point:** t = 0.3770929575 on p(person), selected for maximum F1 on validation and applied as p ≥ t. It is equivalent to d ≥ ln(t/(1−t)) = −0.5019. It is reported for comparison: on the test set, false negatives go from 503 to 336 and false positives from 241 to 399.
   - Choosing it for deployment is an application decision, not part of the port. Its test figures are not independent (§2).
   - The margin also avoids computing a softmax on the chip.
   - In the archived FP32 validation scores one positive image has p exactly equal to t: the F1-optimal threshold coincides with the score of that image. The historical comparison therefore depends on the rule p ≥ t. With p ≥ t that image counts as person (2,496 positives at t); with p > t it would count as no_person (2,495). The conformance rule of this work is d > 0, with ties → no_person.

## 7. Flash and RAM

**Reading corrected after the review.** 479,040 B are the persistent arrays, in flash. 512 B are only the two activation buffers. Workspace, stack and firmware are separate items; the last two can only be measured on the build.

Budget of the ESP32-C3:
- 400 KB of SRAM, 16 KB of which serve as cache;
- 4 MB of flash, with a 1.25 MB application partition in the default Arduino-ESP32 layout.

**Head only, M-hist / H0** (`scripts/estimate_memory.py` → `results/memory_estimate.md`):

| Item | Memory | Bytes | Status |
| --- | --- | --- | --- |
| Persistent arrays: q tables, FP16 scale and y_min, 3 float32 coefficients per edge, knots | Flash (.rodata, read through the cache) | 479,040 | Computed; equal to the built M-hist artifact |
| Activation buffers: two ping-pong float32 buffers sized for the widest layer (64) | SRAM | 512 | Computed |
| Workspace: per-input scalars (SiLU, segment, u, w); 4 B per input if the SiLU of a layer is precomputed | SRAM (stack) | 0 to 256 | Estimated, depends on the implementation |
| Reference vectors for the on-device test (19 vectors, input + h1 + h2 + h3 + logits) | Flash | 8,664 | Estimated, test build only |
| Stack of the inference task | SRAM | — | To measure (`uxTaskGetStackHighWaterMark`) |
| Firmware: code, Arduino-ESP32/ESP-IDF runtime, heap | Flash + SRAM | — | To measure (`pio run -t size`, `ESP.getFreeHeap()`) |

**Margins on the real chip.**
- **SRAM.** The hardware results of lut-kan-ids, on an ESP32-C3 SuperMini with the Arduino runtime, report a free heap of 304,696 B (`results/LUT_KAN_NODIV_K4_L8_hw.json`, field `free_heap_b`). It is the only measured figure for the same chip. The head needs 512 B of buffers plus its workspace and stack: the margin is wide, but the stack is measured in C3-03.
- **Flash.** The tables take 36.5 % of the default 1.25 MB application partition (1,310,720 B). Whether the firmware fits in the remaining 831,680 B is checked on the first build (C3-03).

The same formula gives 271,120 B for S1 (64→16→8→2), 122,592 B for S2 (64→8→2) and 339,792 B for S3 (48→24→12→2). These are recorded only for comparison and are not proposed (§2).

**Full pipeline: an analytical scenario, not a demonstrated feasibility.** These figures come from shapes and data types. No int8 quantisation or block fusion has been implemented or measured.

| Execution scheme (CNN activations, batch 1) | Peak live elements | float32 (B) | int8 (B) |
| --- | --- | --- | --- |
| Operation by operation, full tensors | 327,680 | 1,310,720 | 327,680 |
| Conv + BN + ReLU + pool fused per block | 118,784 | 475,136 | 118,784 |
| Fused blocks, input image streamed by rows | 93,184 | 372,736 | 93,184 |

- **CNN weights with BN folded:** 43,568 weights (the same count as in M-hist) and 136 biases. That is 174,816 B in float32, or about 44,656 B in int8 with per-channel scales.
- **Total flash with the head:** 653,856 B (float32 CNN) or 523,696 B (int8 CNN).
- **Compute:** 28.3 million MAC per image, all emulated in software (no FPU).
- **Feasibility:** a float32 pipeline does not fit in SRAM. An int8 pipeline would also need calibration and new accuracy measurements.

For context, the historical package reports in Python a footprint of 266.8 KiB (CNN + float KAN layers) against 640.7 KiB (CNN + LUT). That is a Python measurement, not an MCU one, but it agrees with the point above: the tables cost memory and save time.

## 8. Approved perimeter

Approved by the supervisor on Nov 24, 2026.

- **Minimum result.** A C++ engine for the M-hist head, implementing contract v2, on the ESP32-C3 SuperMini. The inputs are the precomputed 64-d features of fixed test images, computed in Python and embedded as vectors.
  - The demo reports the two logits [z0, z1] and the margin d = z1 − z0. No probability or softmax is computed on the chip.
  - Conformance uses d > 0, with ties → no_person (§6).
  - Order of work in C3-03: first the 14 public F0 vectors, on the PC and in Wokwi, comparing every layer with Python; then the 19 M-hist vectors, locally.
  - Flash, RAM, stack and latency are measured in Wokwi and in the hardware test.
- **Not claimed.** No end-to-end recognition from images, and no generalisation (the test set is part of validation, §2). The title and the conclusions will say "KAN/LUT head on precomputed features".
- **Kept separate.** The feasibility of the full pipeline remains the analytical scenario of §7. The camera is optional and outside the minimum.

## 9. Decisions on the open questions

Decided by the supervisor on Nov 24, 2026.

1. **Board and toolchain.** The physical board is the ESP32-C3 SuperMini. Platform and toolchain versions are the ones fixed in the C3-03 package.
2. **Outputs and threshold.** The decision rule is d > 0, with ties → no_person. The demo firmware reports only the two logits and the margin; the historical operating point is not computed on the device and remains a comparison in §6.
3. **Private vectors.** The supervisor regenerates the 19 M-hist vectors with the scripts in `c3-02/scripts/`. Weights, M-hist LUTs and any firmware that embeds them stay local and are not published; firmware built with the F0 tables can be public.

## 10. Choices and limits

As a model, I used the historical checkpoint M-hist. It is the only one already trained with the reference architecture, and its provenance is verified: the tables I obtained are identical to those of the previous thesis. 

Non-finite inputs (NaN and infinity) are rejected because they would lead to undefined behavior, risking exceptions/errors and meaningless decisions. 
Limitations: the results do not show that the model generalizes, because the test images are included in the validation on which the model and threshold were chosen. The complete pipeline, from camera to decision, is only estimated. To get feedback on real time, a fedback from direct hardware testing will be needed.

From the IDS engine I take the structure, not the numerical rules. The C++ engine and the header generator will be my work in C3-03.


## 11. Reproducibility, code reuse and assistance

**Reproducibility.**
- Setup and commands are in `c3-02/README.md`.
- The F0 results (r1) are unchanged.
- The M-hist steps need the checkpoint in `private/checkpoint/` (name and SHA-256 in the README) and write their private outputs to `private/c3-02/`, which is git-ignored.
- Only metadata, hashes and aggregate metrics are committed in `results/checkpoint/`.
- Two consecutive runs give identical public results.

**Reused:**
- pykan (model, B-splines);
- lut-kan: `build_lut_for_edges`, `save_lut_npz`, `load_lut_npz`, `pack_dense_layer`, `forward_reference`, `PyKANSingleLayerAdapter`;
- kan-image-iot: `KANImageClassifier`, evaluation preprocessing, dataset;
- the historical package: checkpoint, LUT artifacts and scores, used only as references to compare against;
- lut-kan-ids: studied for §5; its adaptation is the work of C3-03.

None of the upstream files is modified.

**Own work:** the scripts in `c3-02/scripts/`, the choice of the vectors, the tolerance measurement, and this document.

The scripts, the measurements and the first draft of this document were prepared with AI assistance (Claude), following my requests and the supervisor's review.
