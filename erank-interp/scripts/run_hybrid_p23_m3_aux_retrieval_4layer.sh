#!/usr/bin/env bash
# Phase B rescue: hybrid retrieve-then-add with auxiliary retrieval supervision.
# Single seed, 4-layer standard model, p=23, m=3.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
python scripts/run_hybrid_p23_m3_aux_retrieval_4layer.py "$@"
