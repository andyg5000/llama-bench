# llama-bench: Local Inference Optimization Framework

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Find the best llama-server configuration for your model + GPU combination by testing across real use-case profiles.

## What it answers

1. **Is this model+config good for local coding?** Code comprehension, editing, generation, debugging with realistic context
2. **Is this model+config good for agentic use (OpenClaw etc)?** Tool calling, structured output, multi-turn, concurrent requests
3. **Is this model fast and accurate?** Quick accuracy checks + generation/prompt processing speed
4. **What settings work best given my resources?** Compare configs side-by-side with letter grades per profile

## Setup

```bash
cp bench.env.example bench.env
vi bench.env  # set your server IP, model path, GPU name
```

| Variable | Description | Example |
|---|---|---|
| `BENCH_REMOTE_USER` | SSH user | `root` |
| `BENCH_REMOTE_HOST` | SSH host | `192.168.x.x` |
| `BENCH_SERVER_PORT` | llama-server port | `8099` |
| `BENCH_GPU` | GPU identifier | `gb10` |
| `BENCH_MODEL` | Model identifier | `Qwen3.5-35B-A3B-Q8_0` |
| `BENCH_MODEL_PATH` | Path to .gguf on remote | `/models/model.gguf` |
| `BENCH_CHAT_TEMPLATE` | Chat template on remote (optional) | `/models/template.jinja` |
| `BENCH_LLAMA_SERVER` | llama-server binary on remote | `/usr/local/bin/llama-server` |

## Usage

```bash
# Run all experiments (single run)
./run_experiments.sh

# Run with 3 repetitions for mean/stddev
./run_experiments.sh --runs 3

# Run a single experiment
./run_single.sh experiments/001_baseline_f16_262k.env

# Compare all results for a GPU/model
python3 evaluate.py --summarize-all output/gb10/Qwen3.5-35B-A3B-Q8_0/results

# Aggregate multi-run stats
python3 evaluate.py --summarize-runs output/gb10/Qwen3.5-35B-A3B-Q8_0/results 001_baseline_f16_262k
```

## Output

Results are organized by GPU and model:

```
output/
  {gpu}/
    {model}/
      results/       # JSON result files (one per experiment per run)
      experiments/   # Copies of experiment .env files used
```

### Score Card

Each experiment produces a score card:

```
=================================================================
  SCORE CARD: 002_q8_cache_262k
  GPU: gb10 | Model: Qwen3.5-35B-A3B-Q8_0
  Config: ctx=262144 cache=q8_0/q8_0 batch=2048/512 par=1
=================================================================
  Coding:    A  (acc=88% gen=50 t/s avg=12.3s)
  Agentic:   B+ (acc=75% conc=100% wall=8.2s)
  Speed:     A  (acc=92% gen=50 t/s)
  Context:   A  (needle=4/4 pp=1923 t/s)
=================================================================
```

### Comparison Table

`--summarize-all` produces a side-by-side comparison:

```
Experiment                      Ctx  Cache    Batch      Par Coding Agent  Speed    Ctx  Gen t/s  PP t/s
--------------------------------------------------------------------------------------------------------------
001_baseline_f16_262k        262144  f16/f16  2048/512     1      A     B+     A      A    49.9  1930.0
002_q8_cache_262k            262144  q8/q8    2048/512     1      A     B+     A      A    50.1  1923.0
```

## Evaluation Profiles

### Coding
Tests code comprehension, editing, generation, debugging, and optimization. Includes thinking-enabled prompts to measure the cost of chain-of-thought on coding tasks.

### Agentic
Tests tool/function calling, structured JSON output, multi-turn conversations, and system prompt adherence. Includes a concurrent load test (4 parallel requests) to measure throughput under multi-agent use.

### Speed
Quick accuracy sanity checks (math, reasoning, knowledge, instructions). Verifies a config doesn't break basic model capabilities.

### Context
Needle-in-a-haystack tests at 4 context lengths (~500, ~4k, ~20k, ~80k tokens) plus a generation speed benchmark. Measures prompt processing speed scaling and retrieval accuracy.

## Adding Experiments

Create a new `.env` file in `experiments/`:

```bash
# experiments/016_my_test.env
CTX_SIZE=262144
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
BATCH_SIZE=4096
UBATCH_SIZE=1024
PARALLEL=1
GPU_LAYERS=100
EXTRA_FLAGS=""
```

## Testing a Different Model/GPU

Just change the env vars:

```bash
BENCH_GPU=4090 \
BENCH_MODEL=Llama-3-70B-Q4 \
BENCH_REMOTE_HOST=10.0.0.5 \
BENCH_MODEL_PATH=/models/llama3-70b-q4.gguf \
./run_experiments.sh
```

Results go to `output/4090/Llama-3-70B-Q4/`.
