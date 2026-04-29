#!/usr/bin/env bash
# run_all_steps.sh
#
# Run Steps 1-4 of the senior thesis experiment in sequence.
# Must be executed from the project root:
#   bash scripts/run_all_steps.sh
#
# Each step only runs if the previous step succeeded.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate virtual environment.
source "${PROJECT_ROOT}/venv/bin/activate"

# Ensure all imports resolve to the project root.
export PYTHONPATH="${PROJECT_ROOT}"

# Create the log directory before any tee calls.
mkdir -p "${PROJECT_ROOT}/outputs/control_rerun/logs"

LOGS_DIR="${PROJECT_ROOT}/outputs/control_rerun/logs"

echo "============================================================"
echo "  eRank Thesis Experiment — Steps 1–4"
echo "  Project root: ${PROJECT_ROOT}"
echo "  PYTHONPATH:   ${PYTHONPATH}"
echo "============================================================"

# ------------------------------------------------------------
# Step 1+2 (training): train_all_v2.py implements Step 1
# (validation split + checkpoint selection) and records inline
# eRank for Step 4.
# ------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  STEP 1+2 (training): train_all_v2.py"
echo "  Started: $(date)"
echo "------------------------------------------------------------"

python "${PROJECT_ROOT}/scripts/train_all_v2.py" \
    2>&1 | tee "${LOGS_DIR}/train_all_v2.log"

echo "  Finished: $(date)"

# ------------------------------------------------------------
# Step 2 (eRank analysis on best-val checkpoints)
# ------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  STEP 2: run_step2_erank.py"
echo "  Started: $(date)"
echo "------------------------------------------------------------"

python "${PROJECT_ROOT}/scripts/run_step2_erank.py" \
    2>&1 | tee "${LOGS_DIR}/step2_erank.log"

echo "  Finished: $(date)"

# ------------------------------------------------------------
# Step 3 (peak vs final for attention_only_modular_addition)
# ------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  STEP 3: run_step3_peak_vs_final.py"
echo "  Started: $(date)"
echo "------------------------------------------------------------"

python "${PROJECT_ROOT}/scripts/run_step3_peak_vs_final.py" \
    2>&1 | tee "${LOGS_DIR}/step3_peak_vs_final.log"

echo "  Finished: $(date)"

# ------------------------------------------------------------
# Step 4 (training-time eRank from inline data)
# ------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  STEP 4: run_step4_training_time.py"
echo "  Started: $(date)"
echo "------------------------------------------------------------"

python "${PROJECT_ROOT}/scripts/run_step4_training_time.py" \
    2>&1 | tee "${LOGS_DIR}/step4_training_time.log"

echo "  Finished: $(date)"

echo ""
echo "============================================================"
echo "  All steps complete."
echo "  Outputs in: ${PROJECT_ROOT}/outputs/control_rerun/"
echo "============================================================"
