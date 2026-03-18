# llama-bench: Autoresearch for Local LLM Inference

Following [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an LLM-driven experiment loop that automatically finds the best llama-server configuration for your model + GPU.

## How it works

1. Run a seed experiment to establish a baseline
2. An LLM analyzes the results and proposes the next config to try
3. The experiment runs, gets scored, and is committed to git
4. If the score improved: keep the commit. If not: discard it.
5. Repeat until convergence or budget exhausted

Git history becomes the research memory. `results.tsv` is the append-only experiment log the LLM reads to decide what to try next.

## What it optimizes for

1. **Local coding assistant** — prompt ingestion speed, code generation quality
2. **Agentic use (OpenClaw etc)** — tool calling, structured output, concurrent throughput
3. **Speed and accuracy** — generation t/s, prompt processing t/s, correctness
4. **Resource efficiency** — best performance within your memory budget

Each experiment produces a score card with letter grades per profile and a composite score that drives the keep/discard decision.

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

### Autoresearch (recommended)

```bash
# Start the LLM-driven search with a seed experiment
./run_auto.sh --seed experiments/001_baseline_f16_262k.env --memory 128GB

# Custom budget (default 20)
./run_auto.sh --seed experiments/001_baseline_f16_262k.env --budget 30

# Use a different LLM for research decisions
./run_auto.sh --research-host 10.0.0.5 --research-port 8080 --seed experiments/001_baseline_f16_262k.env
```

The researcher creates a git branch (`autoresearch/{gpu}`) and commits each experiment. Improved configs are kept, degraded ones are discarded. The branch history is the research log.

### Manual mode

```bash
# Run all predefined experiments
./run_experiments.sh

# Run with 3 repetitions for mean/stddev
./run_experiments.sh --runs 3

# Run a single experiment
./run_single.sh experiments/001_baseline_f16_262k.env
```

### View results

```bash
# Score card for a single result
python3 evaluate.py --summarize output/gb10/Qwen3.5-35B-A3B-Q8_0/results/001_baseline_f16_262k.json

# Comparison table
python3 evaluate.py --summarize-all output/gb10/Qwen3.5-35B-A3B-Q8_0/results

# Multi-run aggregation
python3 evaluate.py --summarize-runs output/gb10/Qwen3.5-35B-A3B-Q8_0/results 001_baseline_f16_262k

# Raw experiment log
cat output/gb10/Qwen3.5-35B-A3B-Q8_0/results.tsv
```

## Repository structure

This project uses two repos:

- **llama-bench** (this repo) — code, eval prompts, experiment definitions, search space
- **llama-bench-results** (submodule at `output/`) — all results data, organized by GPU/model

The results repo is a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Autoresearch commits/resets happen in the submodule only — the main repo stays clean. Each GPU gets its own branch in the results repo (e.g. `autoresearch/gb10`).

```bash
# Clone with results
git clone --recurse-submodules <repo-url>

# Or init submodule after clone
git submodule update --init
```

### Results layout

```
output/                          # <- git submodule (llama-bench-results)
  {gpu}/
    {model}/
      results/                   # JSON result files
      experiments/               # .env files (kept experiments only in git)
      results.tsv                # Append-only experiment log (LLM reads this)
```

### Setup the results repo

```bash
# Create the results repo on GitHub, then update the submodule URL
git submodule set-url output https://github.com/YOUR_USER/llama-bench-results.git
cd output && git remote set-url origin https://github.com/YOUR_USER/llama-bench-results.git
```

## Key files

| File | Role |
|---|---|
| `program.md` | Research directions and constraints (human-written) |
| `search_space.json` | Valid parameter ranges and constraints |
| `researcher.py` | Autoresearch loop — LLM proposes, run, score, keep/discard |
| `evaluate.py` | Evaluation harness — runs test suites, produces score cards |
| `run_single.sh` | Runs one experiment (start server, evaluate, stop) |
| `run_auto.sh` | Entry point for autoresearch mode |
| `run_experiments.sh` | Manual mode — run predefined experiments |
| `eval_prompts/` | Test suites (coding, agentic, speed, long_context) |
| `experiments/` | Predefined experiment configs |

## Testing a different model/GPU

```bash
BENCH_GPU=4090 \
BENCH_MODEL=Llama-3-70B-Q4 \
BENCH_REMOTE_HOST=10.0.0.5 \
BENCH_MODEL_PATH=/models/llama3-70b-q4.gguf \
./run_auto.sh --seed experiments/001_baseline_f16_262k.env --memory 24GB
```

Results go to `output/4090/Llama-3-70B-Q4/`. Each GPU gets its own autoresearch branch.
