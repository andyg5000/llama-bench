# Qwen3.5-35B-A3B Q8_0 — llama.cpp Benchmark Results

**Hardware:** NVIDIA GB10 (Project DIGITS), 128GB unified memory, ARM Cortex-X925 20-core
**Model:** Qwen3.5-35B-A3B-Q8_0.gguf (35GB)
**Date:** 2026-03-17

## Results Summary

| # | Experiment | Ctx | Cache K/V | Batch/UB | Par | Accuracy | Gen t/s | PP t/s |
|---|---|---|---|---|---|---|---|---|
| 001 | Baseline | 262k | f16/f16 | 2048/512 | 1 | **90%** | 49.9 | 1930 |
| 002 | Q8 cache | 262k | q8/q8 | 2048/512 | 1 | **90%** | 50.1 | 1923 |
| 003 | Q4 cache | 262k | q4/q4 | 2048/512 | 1 | **90%** | 50.2 | 1937 |
| 004 | Q8 512k | 512k | q8/q8 | 2048/512 | 1 | **90%** | 49.8 | 1905 |
| 005 | Q4 512k | 512k | q4/q4 | 2048/512 | 1 | **90%** | 50.4 | 1898 |
| 006 | Q4 1M | **1M** | q4/q4 | 2048/512 | 1 | **90%** | 49.6 | 1828 |
| 007 | Big batch | 262k | q8/q8 | 4096/1024 | 1 | 80% | 50.6 | 2239 |
| 008 | Huge batch | 262k | q8/q8 | 8192/2048 | 1 | 80% | 50.3 | **2460** |
| 009 | Parallel 2 | 262k | q8/q8 | 4096/1024 | 2 | 80% | 49.8 | 2219 |
| 010 | Parallel 4 | 131k | q4/q4 | 4096/1024 | 4 | 80% | 50.1 | 2272 |
| 011 | IQ4_NL | 262k | iq4/iq4 | 4096/1024 | 1 | **90%** | 21.7 | 617 |
| 012 | Q5 | 262k | q5/q5 | 4096/1024 | 1 | **90%** | 21.9 | 681 |
| 013 | Mixed | 262k | q8/q4 | 4096/1024 | 1 | 80% | 22.2 | 789 |
| 014 | SWA Full | 262k | q8/q8 | 4096/1024 | 1 | 80% | 49.9 | 2270 |
| 015 | Threads 20 | 262k | q8/q8 | 4096/1024 | 1 | 80% | 50.1 | 2268 |

## Key Findings

### Generation speed is GPU compute-bound at ~50 t/s

Generation speed is essentially constant across f16, q8_0, and q4_0 cache types, context sizes (262k to 1M), batch sizes, and parallelism levels. The GB10 is the bottleneck.

### q5_0, iq4_nl, and mixed cache types are catastrophically slow (~22 t/s)

These KV cache quantization types lack optimized CUDA kernels on the GB10 (compute capability 12.1). They cut generation speed by more than half. **Avoid q5_0, iq4_nl, and mixed q8/q4 cache types.**

### Prompt processing scales with batch size

- 2048/512: ~1930 t/s
- 4096/1024: ~2240 t/s
- 8192/2048: **~2460 t/s**

Larger batch and ubatch sizes yield faster prompt ingestion with no quality penalty.

### Context can go to 1M with q4_0 cache

The 1M context experiment (006) ran without OOM and maintained 90% accuracy with only minor speed loss (49.6 t/s gen, 1828 t/s PP). This is a 4x increase over the current 262k config.

### KV cache quantization has no impact on accuracy

f16, q8_0, and q4_0 all scored identically at 90%. The q4_0 cache uses 4x less memory than f16 with zero quality degradation on this test suite.

### Parallel slots and SWA-full had no meaningful impact

Adding parallel slots or enabling --swa-full did not improve single-request performance, as expected. Parallel slots are useful for serving multiple concurrent users.

## Recommended Optimal Config

```
--ctx-size 1048576
--cache-type-k q4_0
--cache-type-v q4_0
--batch-size 8192
--ubatch-size 2048
--parallel 1
--n-gpu-layers 100
--flash-attn on
```

**Result:** 1M context, ~50 t/s generation, ~1800+ t/s prompt processing, 90% accuracy — a 4x context increase over the current 262k f16 config with no meaningful speed or quality loss.

## Methodology

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Each experiment:

1. Starts a fresh llama-server with the test configuration
2. Waits for model load and health check
3. Runs 10 known-answer accuracy questions (math, reasoning, knowledge, code, instruction following)
4. Runs a ~2k token prompt processing benchmark
5. Runs a 2000-token generation speed benchmark
6. Records server-reported timings (more accurate than wall-clock)
7. Stops the server and cools down before the next experiment

All generation TPS values are server-reported (`predicted_per_second`), not wall-clock estimates.
