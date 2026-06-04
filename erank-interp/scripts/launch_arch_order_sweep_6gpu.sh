#!/usr/bin/env bash
# scripts/launch_arch_order_sweep_6gpu.sh
#
# Multi-GPU queue-based dispatcher for the balanced A/M architecture-order sweep.
#
# Reads the schedule manifest, atomically pops one schedule per worker, and runs
# scripts/run_arch_order_single.py with each worker pinned to one physical GPU.
#
# Conventions:
#   - GPU IDs are given EXPLICITLY via --gpus 2,3,4,5,6,7 (default 0,1,2,3,4,5).
#     Each worker exports CUDA_VISIBLE_DEVICES=<that physical id>, so the child's
#     CUDA runtime sees only that GPU.
#   - Resume-safe: skips runs whose status.txt is in
#       COMPLETED|GROKKED|MEMORIZED_ONLY|FAILED_TO_FIT|PARTIAL.
#     Pass --rerun_failed to also redo FAILED_ERROR runs.
#   - Atomic job claim via flock on a shared queue file under
#       outputs/arch_order_sweep_6gpu/queue/
#
# Usage:
#   bash scripts/launch_arch_order_sweep_6gpu.sh --gpus 2,3,4,5,6,7
#   [--manifest <path>] [--max_epochs N] [--rerun_failed]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUT_ROOT="outputs/arch_order_sweep_6gpu"
MANIFEST_DEFAULT="$OUT_ROOT/schedule_manifest.csv"
QUEUE_DIR="$OUT_ROOT/queue"
LOGS_DIR="$OUT_ROOT/logs"

GPUS_CSV="0,1,2,3,4,5"
MAX_EPOCHS=50000
RERUN_FAILED=0
MANIFEST="$MANIFEST_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)    MANIFEST="$2"; shift 2;;
    --max_epochs)  MAX_EPOCHS="$2"; shift 2;;
    --gpus)        GPUS_CSV="$2"; shift 2;;
    --rerun_failed) RERUN_FAILED=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# Parse GPU list into an array.
IFS=',' read -r -a GPU_ARRAY <<<"$GPUS_CSV"
NUM_WORKERS=${#GPU_ARRAY[@]}

mkdir -p "$QUEUE_DIR" "$LOGS_DIR" "$OUT_ROOT/runs"

# Record GPU assignment.
{
  echo "launch_time=$(date -Is)"
  echo "physical_gpu_ids=$GPUS_CSV"
  echo "num_workers=$NUM_WORKERS"
  echo "max_epochs=$MAX_EPOCHS"
  echo "manifest=$MANIFEST"
  echo "CUDA_VISIBLE_DEVICES_at_launch=${CUDA_VISIBLE_DEVICES:-<unset>}"
} > "$OUT_ROOT/gpu_assignment.txt"

# Activate venv if not already active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/venv/bin/activate"
  fi
fi

# Build the pending-jobs file by reading the manifest and filtering out already-completed runs.
# Schema: each line = "run_id,schedule,depth"
PENDING_FILE="$QUEUE_DIR/pending.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"
> "$PENDING_FILE"

python - <<'PYEOF' "$MANIFEST" "$OUT_ROOT" "$PENDING_FILE" "$RERUN_FAILED"
import csv, os, sys

manifest, out_root, pending_path, rerun_failed_str = sys.argv[1:]
rerun_failed = bool(int(rerun_failed_str))

DONE_LABELS = {"COMPLETED", "GROKKED", "MEMORIZED_ONLY", "FAILED_TO_FIT", "PARTIAL"}
n_total = 0
n_skip = 0
with open(manifest) as f, open(pending_path, "w") as out:
    reader = csv.DictReader(f)
    for row in reader:
        n_total += 1
        run_id = row["run_id"]
        sched = row["schedule"]
        depth = row["depth"]
        run_dir = os.path.join(out_root, "runs", run_id)
        status_path = os.path.join(run_dir, "status.txt")
        if os.path.exists(status_path):
            with open(status_path) as sp:
                status = sp.read().strip().upper()
            if status in DONE_LABELS:
                n_skip += 1
                continue
            if status == "FAILED_ERROR" and not rerun_failed:
                n_skip += 1
                continue
        out.write(f"{run_id},{sched},{depth}\n")

print(f"manifest={manifest}  total_runs={n_total}  already_done={n_skip}  pending={n_total-n_skip}")
PYEOF

PENDING_COUNT=$(wc -l < "$PENDING_FILE")
echo "[$(date -Is)] pending=$PENDING_COUNT  workers=$NUM_WORKERS  max_epochs=$MAX_EPOCHS"

if [[ "$PENDING_COUNT" -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

# Worker loop: atomically pop one job, run it, repeat until queue empty.
run_worker() {
  local worker_id="$1"
  local gpu_id="$2"   # physical GPU id (e.g. one of 2,3,4,5,6,7)
  local wlog="$LOGS_DIR/worker_${worker_id}_gpu${gpu_id}.log"
  echo "[$(date -Is)] worker $worker_id starting (physical GPU $gpu_id)" | tee -a "$wlog"

  while true; do
    # Atomically pop the first line from the pending file.
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

    local run_id schedule depth
    IFS=',' read -r run_id schedule depth <<<"$job"
    local run_dir="$OUT_ROOT/runs/$run_id"
    mkdir -p "$run_dir/logs"
    local run_log="$run_dir/logs/train.log"

    echo "[$(date -Is)] worker $worker_id physical_GPU=$gpu_id -> $run_id ($schedule, depth $depth)" \
      | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"

    # Run the trainer; force failures to be marked.
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/run_arch_order_single.py \
        --schedule "$schedule" \
        --run_name "$run_id" \
        --depth "$depth" \
        --output_root "$OUT_ROOT" \
        --max_epochs "$MAX_EPOCHS" \
        > "$run_log" 2>&1
    local rc=$?

    if [[ $rc -ne 0 ]]; then
      echo "FAILED_ERROR (rc=$rc)" > "$run_dir/status.txt"
      echo "[$(date -Is)] worker $worker_id $run_id FAILED rc=$rc" \
        | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"
    else
      local final_status
      final_status=$(cat "$run_dir/status.txt" 2>/dev/null || echo UNKNOWN)
      echo "[$(date -Is)] worker $worker_id $run_id done: $final_status" \
        | tee -a "$wlog" | tee -a "$LOGS_DIR/tmux_master.log"
    fi
  done
}

# Spawn one worker per physical GPU listed in --gpus.
for ((w = 0; w < NUM_WORKERS; w++)); do
  run_worker "$w" "${GPU_ARRAY[$w]}" &
done

wait

echo "[$(date -Is)] all workers done. Running aggregator..."
python scripts/aggregate_arch_order_sweep.py --output_root "$OUT_ROOT" \
  2>&1 | tee -a "$LOGS_DIR/tmux_master.log"

echo "[$(date -Is)] sweep complete."
