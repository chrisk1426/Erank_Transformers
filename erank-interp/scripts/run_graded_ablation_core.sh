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
echo "Graded Ablation Core"
echo "=========================================="
python scripts/run_graded_ablation_core.py
