#!/usr/bin/env bash
# Launch all 4 training runs in parallel, one per GPU.
#
# Run assignments:
#   GPU 1 — standard       × modular_addition  (run index 0, long: up to 50k epochs)
#   GPU 2 — standard       × key_value         (run index 1)
#   GPU 3 — attention_only × modular_addition  (run index 2)
#   GPU 4 — attention_only × key_value         (run index 3)
#
# Logs are written to logs/<run_name>.log
# Usage: bash scripts/launch_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

GPUS=(1 2 3 4)
RUN_NAMES=(
  "standard_modular_addition"
  "standard_key_value"
  "attention_only_modular_addition"
  "attention_only_key_value"
)

PIDS=()

for i in 0 1 2 3; do
  GPU=${GPUS[$i]}
  NAME=${RUN_NAMES[$i]}
  LOG="$LOG_DIR/${NAME}.log"

  echo "Launching run $i ($NAME) on GPU $GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/train_all.py" --run $i \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All 4 runs launched. PIDs: ${PIDS[*]}"
echo "Monitor with:"
echo "  tail -f $LOG_DIR/standard_modular_addition.log"
echo "  tail -f $LOG_DIR/standard_key_value.log"
echo "  tail -f $LOG_DIR/attention_only_modular_addition.log"
echo "  tail -f $LOG_DIR/attention_only_key_value.log"
echo ""
echo "Waiting for all runs to finish..."
for PID in "${PIDS[@]}"; do
  wait "$PID" && echo "PID $PID finished OK" || echo "PID $PID FAILED"
done
echo "Done."
