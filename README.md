# ESP32-C3 KAN person detection

Bachelor's thesis project in Computer Engineering (Università eCampus), supervised by Prof. Oleksandr Kuznetsov.

The goal is person detection on an ESP32-C3 microcontroller with a Kolmogorov-Arnold Network (KAN). The KAN edge functions are stored as look-up tables (LUT) and evaluated by a C++ inference engine adapted from the group's LUT-KAN work. Development runs in the Wokwi simulator; the final test runs on real hardware.

Status: in progress.

## Checkpoints

Each checkpoint is a folder plus a git tag on the commit that was delivered. Open the tag to see exactly what was reviewed.

| Id | Folder | Tag | Content | Status |
|---|---|---|---|---|
| C3-01 | [`c3-01/`](c3-01/) | `c3-01` | Minimal ESP32-C3 project in Wokwi: deterministic serial output and smoke test | Approved |
| C3-02 | [`c3-02/`](c3-02/) | `c3-02-r1` (draft), `c3-02-r2` (final) | Model contract: sources and commits, models F0 and M-hist, LUT export format, reference vectors, acceptance criterion, Flash/RAM, proposed perimeter | Approved |

The C++ engine will live in `firmware/` from the next checkpoint on.

The trained checkpoint used from C3-02 on is not published: see [`c3-02/README.md`](c3-02/README.md) for its name, SHA-256 and where to place it locally.

## Upstream sources

This work builds on the research group's repositories; each checkpoint lists the exact commits it uses.

- [kan-image-iot](https://github.com/KuznetsovKarazin/kan-image-iot): KAN image classifier (CNN + KAN head) for person detection
- [lut-kan](https://github.com/KuznetsovKarazin/lut-kan): LUT conversion of KAN layers (format v2)
- [lut-kan-ids](https://github.com/KuznetsovKarazin/lut-kan-ids): LUT-KAN inference engine on ESP32-C3 for intrusion detection

## Authorship

Author: [RodaRoller](https://github.com/RodaRoller). Parts of the code and documentation were prepared with AI assistance (Claude). Each checkpoint README states what is reused and what is own work.
