#!/usr/bin/env bash
# Run a single llama-server experiment on the remote host.
# Usage: ./run_single.sh experiments/001_baseline.env [run_number]
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
MODEL_PATH="${BENCH_MODEL_PATH:?Set BENCH_MODEL_PATH in bench.env or environment}"
CHAT_TEMPLATE="${BENCH_CHAT_TEMPLATE:-}"
LLAMA_SERVER="${BENCH_LLAMA_SERVER:-/usr/local/bin/llama-server}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# Output structure: ./output/{gpu}/{model}/{results,experiments}
OUTPUT_DIR="${SCRIPT_DIR}/output/${GPU}/${MODEL}"
RESULTS_DIR="${OUTPUT_DIR}/results"
EXPERIMENTS_DIR="${OUTPUT_DIR}/experiments"
mkdir -p "$RESULTS_DIR" "$EXPERIMENTS_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment.env> [run_number]"
  exit 1
fi

EXP_FILE="$1"
EXP_NAME="$(basename "$EXP_FILE" .env)"
RUN_NUM="${2:-}"

# Source experiment config
source "$EXP_FILE"

# Copy experiment file to output for reproducibility
cp "$EXP_FILE" "$EXPERIMENTS_DIR/" 2>/dev/null || true

echo "============================================"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU | Model: $MODEL"
[[ -n "$RUN_NUM" ]] && echo "Run: $RUN_NUM"
echo "============================================"
echo "  ctx_size:    ${CTX_SIZE}"
echo "  cache_k:     ${CACHE_TYPE_K}"
echo "  cache_v:     ${CACHE_TYPE_V}"
echo "  batch:       ${BATCH_SIZE}"
echo "  ubatch:      ${UBATCH_SIZE}"
echo "  parallel:    ${PARALLEL}"
echo "  threads:     ${THREADS:-auto}"
echo "  gpu_layers:  ${GPU_LAYERS}"
echo "  extra_flags: ${EXTRA_FLAGS:-}"
echo "============================================"

# Build llama-server command
EXTRA=""
if [[ -n "${THREADS:-}" ]]; then
  EXTRA="--threads ${THREADS}"
fi
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  EXTRA="${EXTRA} ${EXTRA_FLAGS}"
fi

# Build chat template flag if provided
TEMPLATE_FLAG=""
if [[ -n "$CHAT_TEMPLATE" ]]; then
  TEMPLATE_FLAG="--chat-template-file ${CHAT_TEMPLATE}"
fi

# Upload a launcher script to the remote
ssh "$REMOTE" "cat > /tmp/llama-bench-launch.sh" <<LAUNCHER
#!/bin/bash
# Kill any existing experiment server
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 2

export CUDA_VISIBLE_DEVICES=0
exec ${LLAMA_SERVER} \\
  --model ${MODEL_PATH} \\
  --host 0.0.0.0 \\
  --port ${PORT} \\
  --ctx-size ${CTX_SIZE} \\
  --parallel ${PARALLEL} \\
  --cache-type-k ${CACHE_TYPE_K} \\
  --cache-type-v ${CACHE_TYPE_V} \\
  --n-gpu-layers ${GPU_LAYERS} \\
  --flash-attn on \\
  --jinja \\
  ${TEMPLATE_FLAG} \\
  --batch-size ${BATCH_SIZE} \\
  --ubatch-size ${UBATCH_SIZE} \\
  ${EXTRA}
LAUNCHER

ssh "$REMOTE" "chmod +x /tmp/llama-bench-launch.sh"

echo "[*] Starting llama-server..."
ssh -f "$REMOTE" "nohup /tmp/llama-bench-launch.sh > /tmp/llama-bench-server.log 2>&1 &"
sleep 3

# Wait for server to be ready
echo "[*] Waiting for server to load model..."
MAX_WAIT=300
WAITED=0
while true; do
  HEALTH=$(ssh "$REMOTE" "curl -sf http://127.0.0.1:${PORT}/health 2>/dev/null || echo unreachable")
  if echo "$HEALTH" | grep -q '"status":"ok"'; then
    break
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo ""
    echo "[!] Server failed to start within ${MAX_WAIT}s"
    ssh "$REMOTE" "tail -50 /tmp/llama-bench-server.log" || true
    ssh "$REMOTE" "fuser -k ${PORT}/tcp 2>/dev/null" || true
    exit 1
  fi
  printf "."
done
echo ""
echo "[*] Server ready after ${WAITED}s"

# Grab memory breakdown from log
MEMORY_INFO=$(ssh "$REMOTE" "grep memory_breakdown /tmp/llama-bench-server.log 2>/dev/null | tail -2" || echo "N/A")
echo "[*] Memory: ${MEMORY_INFO}"

# Build result filename
if [[ -n "$RUN_NUM" ]]; then
  RESULT_FILE="${RESULTS_DIR}/${EXP_NAME}_run${RUN_NUM}.json"
  RUN_FLAG="--run ${RUN_NUM}"
else
  RESULT_FILE="${RESULTS_DIR}/${EXP_NAME}.json"
  RUN_FLAG=""
fi

# Run evaluation
echo "[*] Running evaluation..."
python3 "${SCRIPT_DIR}/evaluate.py" \
  --host "$REMOTE_HOST" \
  --port "$PORT" \
  --prompts-dir "${SCRIPT_DIR}/eval_prompts" \
  --output "$RESULT_FILE" \
  --experiment "$EXP_NAME" \
  --gpu "$GPU" \
  --model "$MODEL" \
  ${RUN_FLAG} \
  --ctx-size "$CTX_SIZE" \
  --cache-k "$CACHE_TYPE_K" \
  --cache-v "$CACHE_TYPE_V" \
  --batch "$BATCH_SIZE" \
  --ubatch "$UBATCH_SIZE" \
  --parallel "$PARALLEL"

echo "[*] Stopping server..."
ssh "$REMOTE" "fuser -k ${PORT}/tcp 2>/dev/null" || true

echo "[*] Results saved to $RESULT_FILE"
