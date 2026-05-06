#!/bin/bash
# Run 3-layer hybrid m=3 feasibility experiment
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/run_hybrid_3layer_m3.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate venv
if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "venv activated: $(python --version)"
else
    echo "WARNING: no venv found, using system Python"
fi

echo "=========================================="
echo "3-Layer Hybrid m=3 Feasibility Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo ""

python scripts/run_hybrid_3layer_m3.py

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
