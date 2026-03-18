#!/usr/bin/env bash
# Autoresearch loop — LLM-driven experiment search.
#
# Usage:
#   ./run_auto.sh                                          # start with default seed
#   ./run_auto.sh --seed experiments/001_baseline_f16_262k.env  # custom seed
#   ./run_auto.sh --budget 30 --memory 128GB               # custom budget
#   ./run_auto.sh --research-host 10.0.0.5 --research-port 8080  # separate research LLM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load bench config
if [[ -f "${SCRIPT_DIR}/bench.env" ]]; then
  source "${SCRIPT_DIR}/bench.env"
fi

GPU="${BENCH_GPU:?Set BENCH_GPU in bench.env or environment}"
MODEL="${BENCH_MODEL:?Set BENCH_MODEL in bench.env or environment}"

echo "============================================"
echo "  AUTORESEARCH: ${MODEL} on ${GPU}"
echo "  $(date)"
echo "============================================"
echo ""

exec python3 "${SCRIPT_DIR}/researcher.py" "$@"
