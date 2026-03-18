# Research Program

## Goal
Find the optimal llama-server configuration for a given model + GPU combination
across four use cases: coding, agentic, speed, and context handling.

## Approach
- Start with a safe baseline (f16 cache, moderate context, default batch sizes)
- Each experiment changes ONE or TWO parameters from the best-known config
- After establishing baselines for cache types and context sizes, explore batch
  sizes and parallelism
- If an experiment OOMs or errors, back off and note the limit
- Declare convergence when 3+ consecutive experiments show no meaningful improvement

## What "better" means
The primary metric is a weighted composite score:
- Speed profile accuracy + gen t/s (30%)
- Coding profile accuracy + response time (30%)
- Context profile needle accuracy + pp t/s (20%)
- Agentic profile accuracy + concurrent throughput (20%)

A config is "better" if its composite score improves OR if it matches the
current best score with lower resource usage (smaller ctx, less memory).

## Constraints
- Do not use q5_0, iq4_nl, or mixed k/v cache types — these lack optimized
  CUDA kernels on most consumer GPUs and are catastrophically slow
- ubatch_size must be <= batch_size
- parallel > 1 multiplies KV cache memory usage
- Always keep gpu_layers at 100 (full offload) unless testing CPU fallback
