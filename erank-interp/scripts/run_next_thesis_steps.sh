#!/usr/bin/env bash
# scripts/run_next_thesis_steps.sh
#
# Orchestrates the three next thesis steps in parallel on idle GPUs.
# Do NOT use GPU 5 (current hybrid p=113,m=4 seed 1 is running there).
#
#   GPU 3 — causal ablations (fast, CPU-bound mostly)
#   GPU 4 — hybrid m=3 feasibility run (long, ~14h worst case)
#   GPU 3 — advisor figures (after ablations; CPU only, negligible)
#
# Usage:
#   bash scripts/run_next_thesis_steps.sh
#
# Monitor:
#   tail -f outputs/thesis_next_steps_logs/tmux_run.log
#   nvidia-smi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate project venv
if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "[setup] venv activated: $(python --version)"
else
    echo "[setup] WARNING: no venv found, using system Python"
fi

LOG_DIR="outputs/thesis_next_steps_logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "  Thesis Next Steps — Parallel Run"
echo "  Strategy: GPU3=ablations+plots, GPU4=hybrid_m3"
echo "========================================================"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  Python:  $(python3 --version)"
echo "  Git:     $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "  GPU 5:   LEFT UNTOUCHED (current hybrid p=113,m=4 seed 1)"
echo "========================================================"
echo ""

# -----------------------------------------------------------------------
# Step 1: Advisor figures (fast — no GPU needed, run immediately)
# -----------------------------------------------------------------------
echo "[STEP 1] Generating advisor core figures (no GPU)..."
python3 scripts/generate_advisor_figures.py \
    2>&1 | tee "$LOG_DIR/advisor_figures.log"
echo "[STEP 1] DONE advisor figures."
echo ""

# -----------------------------------------------------------------------
# Step 2: Causal ablations on GPU 3 (background)
# -----------------------------------------------------------------------
echo "[STEP 2] Starting causal ablations on GPU 3 (background)..."
CUDA_VISIBLE_DEVICES=3 python3 scripts/run_causal_ablations.py \
    2>&1 | tee "$LOG_DIR/causal_ablations.log" &
ABLATION_PID=$!
echo "  Ablation PID: $ABLATION_PID"
echo ""

# -----------------------------------------------------------------------
# Step 3: Hybrid m=3 feasibility on GPU 4 (foreground — long running)
# -----------------------------------------------------------------------
echo "[STEP 3] Starting hybrid m=3 feasibility run on GPU 4..."
echo "  This may run for up to ~14h. Monitor with:"
echo "    tail -f $LOG_DIR/hybrid_m3.log"
echo ""
CUDA_VISIBLE_DEVICES=4 python3 scripts/run_hybrid_m3_feasibility.py \
    2>&1 | tee "$LOG_DIR/hybrid_m3.log" &
HYBRID_PID=$!
echo "  Hybrid m=3 PID: $HYBRID_PID"
echo ""

# -----------------------------------------------------------------------
# Wait for ablations (should finish in minutes)
# -----------------------------------------------------------------------
echo "Waiting for causal ablations to complete..."
wait $ABLATION_PID
ABLATION_EXIT=$?
if [ $ABLATION_EXIT -eq 0 ]; then
    echo "[STEP 2] DONE causal ablations (exit 0)."
else
    echo "[STEP 2] FAILED causal ablations (exit $ABLATION_EXIT). Check $LOG_DIR/causal_ablations.log"
fi
echo ""

# -----------------------------------------------------------------------
# Wait for hybrid m=3
# -----------------------------------------------------------------------
echo "Waiting for hybrid m=3 feasibility run to complete..."
wait $HYBRID_PID
HYBRID_EXIT=$?
if [ $HYBRID_EXIT -eq 0 ]; then
    echo "[STEP 3] DONE hybrid m=3 (exit 0)."
else
    echo "[STEP 3] FAILED hybrid m=3 (exit $HYBRID_EXIT). Check $LOG_DIR/hybrid_m3.log"
fi
echo ""

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
echo "========================================================"
echo "  All steps complete."
echo "  Outputs:"
echo "    outputs/advisor_core_figures/"
echo "    outputs/causal_ablation_core/"
echo "    outputs/hybrid_simplified_m3_standard_seed0/"
echo "  Logs:"
echo "    $LOG_DIR/"
echo "========================================================"
echo "  GPU 5 hybrid p=113,m=4 seed 1: still running (untouched)"
echo "  Check with: tmux attach -t thesis_additions"
echo "========================================================"
