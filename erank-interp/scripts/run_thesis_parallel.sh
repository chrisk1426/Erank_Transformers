#!/usr/bin/env bash
# scripts/run_thesis_parallel.sh
#
# Queue-based parallel launcher: 1 job per GPU, 3 GPUs running simultaneously.
# Each GPU processes its assigned jobs sequentially — no GPU contention.
#
# Strategy:
#   - 3 GPU queues (GPUs 3, 4, 5), 6 jobs each = 18 total
#   - attention_only runs capped at 15000 epochs (they don't learn, no point
#     running all 50k)
#   - Jobs distributed so each GPU gets a mix of fast and slow tasks
#   - Aggregation runs once all 18 jobs finish
#
# Expected wall time: ~5-6 hours
#
# Launch inside tmux:
#   tmux new-session -d -s thesis_additions \
#     "cd /scratch/f006j44/erank/erank-interp && \
#      source venv/bin/activate && \
#      mkdir -p outputs/thesis_additions && \
#      bash scripts/run_thesis_parallel.sh 2>&1 | tee outputs/thesis_additions/tmux_run.log"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/thesis_additions"

cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
    fi
fi

echo "========================================================"
echo "  Thesis Additions — Queue-Based Parallel Run"
echo "  Strategy: 1 job per GPU, 3 GPUs, queue of 6 jobs each"
echo "========================================================"
echo "  Date:       $(date)"
echo "  Host:       $(hostname)"
echo "  Python:     $(python --version 2>&1)"
echo "  Git commit: $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Output dir: ${OUTPUT_DIR}"
echo "========================================================"

mkdir -p "${OUTPUT_DIR}/aggregate"

# ---------------------------------------------------------------------------
# Clean up any partial logs from previous aborted runs
# ---------------------------------------------------------------------------
echo "  Cleaning partial logs from previous runs..."
find "${OUTPUT_DIR}" -name "log.txt" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Job queues: each GPU gets 6 jobs, defined as "seed model task max_epochs"
# attention_only jobs are capped at 15000 epochs.
# Distributed to balance load (mix fast/slow across GPUs).
# ---------------------------------------------------------------------------

MAX_FULL=50000
MAX_ATTN=15000   # attn-only never learns past this; saves ~3.5 hrs per run

# GPU 3 queue (6 jobs)
GPU3_JOBS=(
    "0 standard       key_value            ${MAX_FULL}"
    "1 standard       key_value            ${MAX_FULL}"
    "2 standard       key_value            ${MAX_FULL}"
    "0 attention_only key_value            ${MAX_FULL}"
    "1 attention_only key_value            ${MAX_FULL}"
    "2 attention_only key_value            ${MAX_FULL}"
)

# GPU 4 queue (6 jobs)
GPU4_JOBS=(
    "0 standard       modular_addition     ${MAX_FULL}"
    "1 standard       modular_addition     ${MAX_FULL}"
    "2 standard       modular_addition     ${MAX_FULL}"
    "0 attention_only modular_addition     ${MAX_ATTN}"
    "1 attention_only modular_addition     ${MAX_ATTN}"
    "2 attention_only modular_addition     ${MAX_ATTN}"
)

# GPU 5 queue (6 jobs)
GPU5_JOBS=(
    "0 standard       hybrid_retrieve_add  ${MAX_FULL}"
    "1 standard       hybrid_retrieve_add  ${MAX_FULL}"
    "2 standard       hybrid_retrieve_add  ${MAX_FULL}"
    "0 attention_only hybrid_retrieve_add  ${MAX_ATTN}"
    "1 attention_only hybrid_retrieve_add  ${MAX_ATTN}"
    "2 attention_only hybrid_retrieve_add  ${MAX_ATTN}"
)

echo "  GPU 3 → all key_value runs (6 × fast ~5 min each)"
echo "  GPU 4 → all modular_addition runs (3 × std ~40 min + 3 × attn capped at 15k)"
echo "  GPU 5 → all hybrid_retrieve_add runs (3 × std + 3 × attn capped at 15k)"
echo "  Attn-only epoch cap: ${MAX_ATTN}"
echo "========================================================"
echo ""

# ---------------------------------------------------------------------------
# Function: run one GPU queue sequentially
# ---------------------------------------------------------------------------
run_gpu_queue() {
    local gpu=$1
    shift
    local jobs=("$@")

    echo "[GPU ${gpu}] Starting queue of ${#jobs[@]} jobs"

    for job in "${jobs[@]}"; do
        read -r seed model task max_ep <<< "${job}"
        run_name="seed${seed}_${model}_${task}"
        run_dir="${OUTPUT_DIR}/seed_${seed}/${model}_${task}"
        mkdir -p "${run_dir}"
        log_file="${run_dir}/log.txt"

        echo "[GPU ${gpu}] START: ${run_name}  (max_epochs=${max_ep})"
        start_ts=$(date +%s)

        CUDA_VISIBLE_DEVICES=${gpu} python scripts/run_single_experiment.py \
            --seed "${seed}" \
            --model_type "${model}" \
            --task "${task}" \
            --max_epochs "${max_ep}" \
            > "${log_file}" 2>&1

        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        echo "[GPU ${gpu}] DONE:  ${run_name}  (${elapsed}s)"
    done

    echo "[GPU ${gpu}] Queue complete."
}

# ---------------------------------------------------------------------------
# Launch 3 GPU queues in parallel (background subshells)
# ---------------------------------------------------------------------------
run_gpu_queue 3 "${GPU3_JOBS[@]}" &
PID3=$!

run_gpu_queue 4 "${GPU4_JOBS[@]}" &
PID4=$!

run_gpu_queue 5 "${GPU5_JOBS[@]}" &
PID5=$!

echo "  GPU queues running in background (PIDs: ${PID3} ${PID4} ${PID5})"
echo "  Monitor progress:"
echo "    grep 'DONE\|START' ${OUTPUT_DIR}/tmux_run.log"
echo "    tail -f ${OUTPUT_DIR}/seed_0/standard_modular_addition/log.txt"
echo ""

# ---------------------------------------------------------------------------
# Wait for all 3 queues
# ---------------------------------------------------------------------------
FAILED=0
for pid in ${PID3} ${PID4} ${PID5}; do
    wait "${pid}" || FAILED=$((FAILED+1))
done

echo ""
echo "========================================================"
echo "  All GPU queues finished.  Failures: ${FAILED}"
echo "  Date: $(date)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
echo ""
echo "  Running aggregation..."
python scripts/aggregate_only.py

echo ""
echo "========================================================"
echo "  Complete."
echo "  Report: ${OUTPUT_DIR}/aggregate/thesis_additions_report.md"
echo "========================================================"
