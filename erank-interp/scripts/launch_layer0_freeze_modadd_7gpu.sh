#!/usr/bin/env bash
# scripts/launch_layer0_freeze_modadd_7gpu.sh
#
# Queue-based dispatcher for the layer-0 freezing modular-addition experiment.
# 12 jobs (4 variants × 3 seeds) over an explicit list of physical GPU IDs.
#
# Default --gpus 1,2,3,4,5,6,7 per the handoff doc ("Do not use GPU 0.").
# Each worker pins its job to one physical GPU via CUDA_VISIBLE_DEVICES.
#
# Resume-safe: skips runs whose status.txt is one of
#   COMPLETED|GROKKED|MEMORIZED_ONLY|FAILED_TO_FIT|PARTIAL
# FAILED_ERROR runs are skipped unless --rerun_failed is passed.
# RUNNING is treated as in-flight; if killed before status.txt is rewritten,
# the trainer overwrites everything on restart.
#
# Usage:
#   bash scripts/launch_layer0_freeze_modadd_7gpu.sh
#     [--gpus 1,2,3,4,5,6,7] [--max_epochs 20000] [--rerun_failed]
#     [--checkpoint_every 10] [--full_checkpoint_every 100]
#
# After all workers drain, calls scripts/analyze_layer0_freeze_modadd.py.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUT_ROOT="outputs/layer0_freeze_modadd"
QUEUE_DIR="$OUT_ROOT/queue"
LOGS_DIR="$OUT_ROOT/logs"

GPUS_CSV="1,2,3,4,5,6,7"
MAX_EPOCHS=20000
CKPT_EVERY=10
FULL_CKPT_EVERY=100
RERUN_FAILED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)                 GPUS_CSV="$2"; shift 2;;
    --max_epochs)           MAX_EPOCHS="$2"; shift 2;;
    --checkpoint_every)     CKPT_EVERY="$2"; shift 2;;
    --full_checkpoint_every) FULL_CKPT_EVERY="$2"; shift 2;;
    --rerun_failed)         RERUN_FAILED=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

IFS=',' read -r -a GPU_ARRAY <<<"$GPUS_CSV"
NUM_WORKERS=${#GPU_ARRAY[@]}

mkdir -p "$QUEUE_DIR" "$LOGS_DIR" "$OUT_ROOT"

{
  echo "launch_time=$(date -Is)"
  echo "physical_gpu_ids=$GPUS_CSV"
  echo "num_workers=$NUM_WORKERS"
  echo "max_epochs=$MAX_EPOCHS"
  echo "checkpoint_every=$CKPT_EVERY"
  echo "full_checkpoint_every=$FULL_CKPT_EVERY"
  echo "CUDA_VISIBLE_DEVICES_at_launch=${CUDA_VISIBLE_DEVICES:-<unset>}"
} > "$OUT_ROOT/gpu_assignment.txt"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/venv/bin/activate"
fi

# Build pending queue: every (variant, seed) whose status.txt is not terminal.
PENDING_FILE="$QUEUE_DIR/pending.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"
> "$PENDING_FILE"

python - <<PYEOF "$OUT_ROOT" "$PENDING_FILE" "$RERUN_FAILED"
import os, sys
out_root, pending_path, rerun_failed_s = sys.argv[1:]
rerun_failed = bool(int(rerun_failed_s))
DONE = {"COMPLETED","GROKKED","MEMORIZED_ONLY","FAILED_TO_FIT","PARTIAL"}
variants = [
    ("normal",       "false","false"),
    ("freeze_A0",    "true", "false"),
    ("freeze_M0",    "false","true"),
    ("freeze_A0_M0", "true", "true"),
]
seeds = [0, 1, 2]
n_total = n_skip = 0
with open(pending_path, "w") as out:
    for v, fa, fm in variants:
        for s in seeds:
            n_total += 1
            run_name = f"{v}_seed{s}"
            sp = os.path.join(out_root, run_name, "status.txt")
            if os.path.exists(sp):
                with open(sp) as f:
                    status = f.read().strip().upper()
                if status in DONE:
                    n_skip += 1; continue
                if status == "FAILED_ERROR" and not rerun_failed:
                    n_skip += 1; continue
            out.write(f"{run_name},{v},{s},{fa},{fm}\n")
print(f"queue total={n_total} done={n_skip} pending={n_total-n_skip}")
PYEOF

PENDING_COUNT=$(wc -l < "$PENDING_FILE")
echo "[$(date -Is)] pending=$PENDING_COUNT  workers=$NUM_WORKERS  max_epochs=$MAX_EPOCHS"

if [[ "$PENDING_COUNT" -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local wlog="$LOGS_DIR/worker_${worker_id}_gpu${gpu_id}.log"
  echo "[$(date -Is)] worker $worker_id starting (physical GPU $gpu_id)" | tee -a "$wlog"

  while true; do
    local job=""
    job="$(flock "$LOCK_FILE" -c "
      if [[ -s '$PENDING_FILE' ]]; then
        head -n 1 '$PENDING_FILE'
        tail -n +2 '$PENDING_FILE' > '$PENDING_FILE.tmp' && mv '$PENDING_FILE.tmp' '$PENDING_FILE'
      fi
    ")"
    if [[ -z "$job" ]]; then
      echo "[$(date -Is)] worker $worker_id: queue empty, exiting" | tee -a "$wlog"
      return 0
    fi

    local run_name variant seed fa fm
    IFS=',' read -r run_name variant seed fa fm <<<"$job"
    local run_dir="$OUT_ROOT/$run_name"
    mkdir -p "$run_dir/logs"
    local run_log="$run_dir/logs/train.log"

    echo "[$(date -Is)] worker $worker_id phys_GPU=$gpu_id -> $run_name ($variant seed=$seed fa=$fa fm=$fm)" \
      | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"

    local freeze_flags=""
    [[ "$fa" == "true" ]] && freeze_flags="$freeze_flags --freeze-a0"
    [[ "$fm" == "true" ]] && freeze_flags="$freeze_flags --freeze-m0"

    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/run_layer0_freeze_modadd.py \
        --variant "$variant" --seed "$seed" $freeze_flags \
        --run-name "$run_name" \
        --output-root "$OUT_ROOT" \
        --max-epochs "$MAX_EPOCHS" \
        --checkpoint-every "$CKPT_EVERY" \
        --full-checkpoint-every "$FULL_CKPT_EVERY" \
        > "$run_log" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
      echo "FAILED_ERROR (rc=$rc)" > "$run_dir/status.txt"
      echo "[$(date -Is)] worker $worker_id $run_name FAILED rc=$rc" \
        | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"
    else
      local final
      final=$(cat "$run_dir/status.txt" 2>/dev/null || echo UNKNOWN)
      echo "[$(date -Is)] worker $worker_id $run_name done: $final" \
        | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"
    fi
  done
}

for ((w = 0; w < NUM_WORKERS; w++)); do
  run_worker "$w" "${GPU_ARRAY[$w]}" &
done
wait

echo "[$(date -Is)] all workers done. Running analyzer..."
python scripts/analyze_layer0_freeze_modadd.py --output-root "$OUT_ROOT" \
  2>&1 | tee -a "$LOGS_DIR/tmux_master.log"
echo "[$(date -Is)] experiment complete."
