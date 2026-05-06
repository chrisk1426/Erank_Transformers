#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "venv activated: $(python --version)"
else
    echo "WARNING: no venv found, using system Python"
fi

echo "=========================================="
echo "Hybrid small depth test: 4-layer"
echo "=========================================="
python scripts/run_hybrid_small_depth_test.py --n_layers 4
