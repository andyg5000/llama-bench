#!/usr/bin/env bash
# Run all experiments sequentially, inspired by karpathy/autoresearch.
# Each experiment: start server -> wait for ready -> evaluate -> stop server.
#
# Usage:
#   ./run_experiments.sh                    # single run of all experiments
#   ./run_experiments.sh --runs 3           # 3 runs per experiment for mean/stddev
#   ./run_experiments.sh path/to/exps/      # use custom experiments dir
#   ./run_experiments.sh --runs 3 path/to/  # both
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load bench config (bench.env or env vars)
if [[ -f "${SCRIPT_DIR}/bench.env" ]]; then
  source "${SCRIPT_DIR}/bench.env"
fi

# Required configuration
REMOTE_USER="${BENCH_REMOTE_USER:?Set BENCH_REMOTE_USER in bench.env or environment}"
REMOTE_HOST="${BENCH_REMOTE_HOST:?Set BENCH_REMOTE_HOST in bench.env or environment}"
PORT="${BENCH_SERVER_PORT:-8099}"
GPU="${BENCH_GPU:?Set BENCH_GPU in bench.env or environment}"
MODEL="${BENCH_MODEL:?Set BENCH_MODEL in bench.env or environment}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# Parse args
NUM_RUNS=1
EXPERIMENTS_DIR="${SCRIPT_DIR}/experiments"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    *)
      EXPERIMENTS_DIR="$1"
      shift
      ;;
  esac
done

OUTPUT_DIR="${SCRIPT_DIR}/output/${GPU}/${MODEL}"
RESULTS_DIR="${OUTPUT_DIR}/results"

echo "============================================"
echo "  llama-bench: ${MODEL}"
echo "  Target: ${GPU}"
echo "  Runs per experiment: ${NUM_RUNS}"
echo "  $(date)"
echo "============================================"
echo ""

# Make sure no experiment server is running
echo "[*] Ensuring clean state..."
ssh "$REMOTE" "fuser -k ${PORT}/tcp 2>/dev/null; sleep 2" || true

# Check that no other llama-server is using significant VRAM
RUNNING=$(ssh "$REMOTE" "ps aux | grep llama-server | grep -v grep" || true)
if [[ -n "$RUNNING" ]]; then
  echo "[!] WARNING: Other llama-server instances are running:"
  echo "$RUNNING"
  echo "[!] These may compete for VRAM. Consider stopping them first."
  echo "[!] Continuing in 5 seconds..."
  sleep 5
fi

TOTAL=0
PASSED=0
FAILED=0

for exp in "$EXPERIMENTS_DIR"/*.env; do
  [[ -f "$exp" ]] || continue
  EXP_NAME="$(basename "$exp" .env)"

  for RUN in $(seq 1 "$NUM_RUNS"); do
    TOTAL=$((TOTAL + 1))

    if [[ "$NUM_RUNS" -gt 1 ]]; then
      echo ""
      echo ">>> Experiment: $EXP_NAME (run $RUN/$NUM_RUNS)"
      echo ""
      RUN_ARG="$RUN"
    else
      echo ""
      echo ">>> Experiment: $EXP_NAME"
      echo ""
      RUN_ARG=""
    fi

    if "$SCRIPT_DIR/run_single.sh" "$exp" $RUN_ARG; then
      PASSED=$((PASSED + 1))
    else
      echo "[!] Experiment $EXP_NAME FAILED"
      FAILED=$((FAILED + 1))
      ssh "$REMOTE" "fuser -k ${PORT}/tcp 2>/dev/null; sleep 2" || true
    fi

    # Cool-down between experiments
    echo "[*] Cooling down for 10s..."
    sleep 10
  done
done

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Total: $TOTAL  Passed: $PASSED  Failed: $FAILED"
echo "============================================"
echo ""

# Final comparison table
echo ">>> Score Card Comparison:"
python3 "$SCRIPT_DIR/evaluate.py" --summarize-all "$RESULTS_DIR"

# If multi-run, show aggregated stats per experiment
if [[ "$NUM_RUNS" -gt 1 ]]; then
  echo ">>> Per-Experiment Aggregates:"
  for exp in "$EXPERIMENTS_DIR"/*.env; do
    [[ -f "$exp" ]] || continue
    EXP_NAME="$(basename "$exp" .env)"
    python3 "$SCRIPT_DIR/evaluate.py" --summarize-runs "$RESULTS_DIR" "$EXP_NAME"
  done
fi
