| Item | Memory | Bytes | Status |
|---|---|---|---|
| Persistent arrays: q tables, FP16 scale/y_min, 3 float32 coefficients per edge, knots | flash (.rodata, read through the cache) | 479 040 | computed |
| Activation buffers: two ping-pong float32 buffers of the widest layer | SRAM | 512 | computed |
| Workspace: per-input scalars (SiLU, segment, u, w); 4 B per input if SiLU is precomputed per layer | SRAM (stack) | 0 to 256 | estimated |
| Reference vectors for the on-device test (19 x (input + h1 + h2 + h3 + logits), float32) | flash | 8 664 | estimated (test build only) |
| Stack of the inference task | SRAM | - | to measure (uxTaskGetStackHighWaterMark) |
| Firmware: code, Arduino-ESP32/ESP-IDF runtime, heap | flash + SRAM | - | to measure (pio run -t size, ESP.getFreeHeap()) |

| Id | Head | Edges | Flash (B) | RAM, tables in flash (B) | RAM, tables copied to SRAM (B) |
|---|---|---|---|---|---|
| H0 | 64→24→16→8→2 | 2 064 | 479 040 | 512 | 479 552 |
| S1 | 64→16→8→2 | 1 168 | 271 120 | 512 | 271 632 |
| S2 | 64→8→2 | 528 | 122 592 | 512 | 123 104 |
| S3 | 48→24→12→2 | 1 464 | 339 792 | 384 | 340 176 |

| Scheme | Peak live elements | float32 (B) | int8 (B) |
|---|---|---|---|
| op_by_op | 327 680 | 1 310 720 | 327 680 |
| fused_blocks | 118 784 | 475 136 | 118 784 |
| fused_blocks_streamed_input | 93 184 | 372 736 | 93 184 |

CNN parameters (BN folded): 43 568 weights + 136 biases; float32 174 816 B, int8 44 656 B; 28 344 320 MAC per image.
Pipeline flash (CNN + H0): float32 653 856 B, int8 CNN 523 696 B; one 128x128 RGB test image 49 152 B.
 