#!/usr/bin/env bash
# scripts/run_novel_experiments_queue.sh
#
# Master dispatcher for the Part A + Part B experiments described in
# claude_code_novel_architecture_probe_experiments.md.
#
# Runs a job queue on 2 GPUs (default: 4 and 6, override with WORKER_GPUS).
# Probes + FFN-head are queued FIRST so cheaper results land within a few
# hours; the 7 architecture-matrix runs follow.
#
# Usage:
#   bash scripts/run_novel_experiments_queue.sh
#   WORKER_GPUS="4,6" bash scripts/run_novel_experiments_queue.sh
#
# Under tmux:
#   tmux new-session -d -s novel_experiments \
#       "bash scripts/run_novel_experiments_queue.sh 2>&1 | tee outputs/novel_experiments_master.log"

set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

WORKER_GPUS_RAW="${WORKER_GPUS:-4,6}"
IFS=',' read -ra WORKERS <<< "${WORKER_GPUS_RAW}"
ARCH_OUT_ROOT="outputs/modadd_arch_matrix_7gpu"
PROBE_OUT="outputs/hybrid_probe_diagnostics"
FFN_OUT_ROOT="outputs/hybrid_ffn_head"
mkdir -p "${ARCH_OUT_ROOT}/logs" "${PROBE_OUT}/logs" "${FFN_OUT_ROOT}/logs"

# Queue is a temp file with one job per line. Format:
#   <log_path>|||<command>
QUEUE_FILE=$(mktemp /tmp/novel_queue.XXXXXX)
LOCK_FILE="${QUEUE_FILE}.lock"
touch "${LOCK_FILE}"

queue_job () {
  local logp="$1"; local cmd="$2"
  printf '%s|||%s\n' "${logp}" "${cmd}" >> "${QUEUE_FILE}"
}

# --- Part B (cheap, queued first) ----------------------------------------
queue_job "${PROBE_OUT}/logs/probe_diagnostics.log" \
  "python analysis/run_hybrid_probe_diagnostics.py --out_dir ${PROBE_OUT}"

queue_job "${FFN_OUT_ROOT}/logs/hybrid_p23_m3_4layer_ffn_head_seed0.log" \
  "python scripts/run_hybrid_ffn_head.py --output_root ${FFN_OUT_ROOT} --run_name hybrid_p23_m3_4layer_ffn_head_seed0"

# --- Part A: 7 modular-addition arch runs --------------------------------
ARCH_NAMES=(
  "modadd_4layer_standard_AMAMAMAM"
  "modadd_4layer_attention_only_AAAA"
  "modadd_4layer_mlp_only_MMMM"
  "modadd_4layer_early_attention_AMMM"
  "modadd_4layer_late_attention_MMMA"
  "modadd_4layer_route_then_process_AAMM"
  "modadd_4layer_process_then_route_MMAA"
)
ARCH_SCHEDULES=(
  "AM,AM,AM,AM"
  "A,A,A,A"
  "M,M,M,M"
  "A,M,M,M"
  "M,M,M,A"
  "A,A,M,M"
  "M,M,A,A"
)
for i in "${!ARCH_NAMES[@]}"; do
  name="${ARCH_NAMES[$i]}"
  sched="${ARCH_SCHEDULES[$i]}"
  queue_job "${ARCH_OUT_ROOT}/logs/${name}.log" \
    "python scripts/run_modadd_arch.py --schedule ${sched} --run_name ${name} --output_root ${ARCH_OUT_ROOT}"
done

echo "[$(date '+%F %T')] Queued $(wc -l < "${QUEUE_FILE}") jobs on workers: ${WORKERS[*]}"
echo "[$(date '+%F %T')] Queue file: ${QUEUE_FILE}"

# --- Worker function ------------------------------------------------------
# Atomically pops one line from the queue. Echoes line to stdout.
# Returns exit 0 if a job was popped, exit 1 if queue is empty.
pop_job () {
  flock -x "${LOCK_FILE}" -c "
    line=\$(head -n1 '${QUEUE_FILE}')
    if [ -z \"\$line\" ]; then exit 1; fi
    sed -i '1d' '${QUEUE_FILE}'
    printf '%s\n' \"\$line\"
  "
}

run_worker () {
  local gpu="$1"
  while true; do
    local line
    line=$(pop_job)
    local rc=$?
    if [ "${rc}" -ne 0 ] || [ -z "${line}" ]; then
      echo "[$(date '+%F %T')] worker gpu=${gpu}: queue empty, exiting."
      return 0
    fi
    local logp="${line%%|||*}"
    local cmd="${line#*|||}"
    mkdir -p "$(dirname "${logp}")"
    echo "[$(date '+%F %T')] worker gpu=${gpu} START -> ${logp}"
    echo "  cmd: CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" > "${logp}" 2>&1
    local jrc=$?
    echo "[$(date '+%F %T')] worker gpu=${gpu} END exit=${jrc} log=${logp}"
  done
}

# --- Launch one worker per GPU --------------------------------------------
WPIDS=()
for gpu in "${WORKERS[@]}"; do
  run_worker "${gpu}" &
  WPIDS+=($!)
done
echo "[$(date '+%F %T')] worker PIDs: ${WPIDS[*]}"

# Wait for all workers to drain the queue.
for pid in "${WPIDS[@]}"; do
  wait "${pid}" || true
done
echo "[$(date '+%F %T')] All workers finished."

# --- Aggregate after queue drains ----------------------------------------
echo "[$(date '+%F %T')] Aggregating arch matrix..."
python scripts/aggregate_modadd_arch_matrix.py --output_root "${ARCH_OUT_ROOT}" \
    > "${ARCH_OUT_ROOT}/logs/aggregate.log" 2>&1 || true
echo "[$(date '+%F %T')] Writing consolidated Part-D report..."
python scripts/aggregate_novel_experiments.py \
    --arch_root "${ARCH_OUT_ROOT}" \
    --probe_root "${PROBE_OUT}" \
    --ffn_root "${FFN_OUT_ROOT}" \
    > outputs/novel_experiments_aggregate.log 2>&1 || true

# Cleanup queue tempfile
rm -f "${QUEUE_FILE}" "${LOCK_FILE}"
echo "[$(date '+%F %T')] DONE."
