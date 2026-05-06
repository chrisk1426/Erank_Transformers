#!/usr/bin/env bash
# scripts/run_thesis_additions.sh
#
# Run all thesis-additions experiments inside a persistent process.
# Designed to be launched inside tmux so the user can disconnect from SSH.
#
# Usage (from erank-interp directory):
#   bash scripts/run_thesis_additions.sh
#
# Or launched via tmux (from project root):
#   tmux new-session -d -s thesis_additions \
#     "cd /scratch/f006j44/erank/erank-interp && \
#      source venv/bin/activate && \
#      mkdir -p outputs/thesis_additions && \
#      bash scripts/run_thesis_additions.sh 2>&1 | tee outputs/thesis_additions/tmux_run.log"

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/thesis_additions"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

cd "${PROJECT_ROOT}"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
        echo "[setup] Activating venv..."
        source "${PROJECT_ROOT}/venv/bin/activate"
    else
        echo "[setup] WARNING: No venv found at ${PROJECT_ROOT}/venv. Proceeding with current Python."
    fi
fi

# ---------------------------------------------------------------------------
# Pre-flight info
# ---------------------------------------------------------------------------

echo "========================================================"
echo "  Thesis Additions Experiment Run"
echo "========================================================"
echo "  Date:        $(date)"
echo "  Host:        $(hostname)"
echo "  Python:      $(python --version 2>&1)"
echo "  Venv:        ${VIRTUAL_ENV:-none}"
echo "  Project:     ${PROJECT_ROOT}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Git commit:  $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Create output directories before any tee/redirect
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}/aggregate"

# ---------------------------------------------------------------------------
# Phase 1: Smoke test (fast, ~100 epochs)
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo "  Phase 1: Smoke test (verify new hybrid task works)"
echo "========================================================"

python - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())

# Test 1: hybrid data generation
print("[smoke] Testing hybrid data generation...")
from data.hybrid_retrieve_add import generate_hybrid_data, sanity_check_labels
inputs, labels = generate_hybrid_data(p=113, num_kv_pairs=4, num_examples=200, seed=0)
sanity_check_labels(inputs, labels, p=113, num_kv_pairs=4, n_check=50)
print("[smoke] hybrid data generation: PASSED")

# Test 2: hybrid 3-way split
print("[smoke] Testing hybrid 3-way split...")
from data.splits import get_hybrid_dataloaders_3way
train_l, val_l, test_l, meta = get_hybrid_dataloaders_3way(
    p=113, num_kv_pairs=4, num_examples=500, seed=0, batch_size=64
)
print(f"[smoke] split: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}")
print("[smoke] hybrid 3-way split: PASSED")

# Test 3: hybrid model creation
print("[smoke] Testing hybrid model creation...")
from models.transformer import create_standard_transformer, create_attention_only_transformer
import torch
std_model  = create_standard_transformer("hybrid_retrieve_add",
    cfg_overrides={"hybrid_p": 113, "hybrid_num_kv_pairs": 4}, seed=0)
attn_model = create_attention_only_transformer("hybrid_retrieve_add",
    cfg_overrides={"hybrid_p": 113, "hybrid_num_kv_pairs": 4}, seed=0)
print(f"[smoke] standard cfg: n_ctx={std_model.cfg.n_ctx} d_vocab={std_model.cfg.d_vocab} d_vocab_out={std_model.cfg.d_vocab_out}")
assert std_model.cfg.n_ctx == 12 and std_model.cfg.d_vocab == 119 and std_model.cfg.d_vocab_out == 113
print("[smoke] model creation: PASSED")

# Test 4: forward pass
print("[smoke] Testing forward pass...")
for batch, _ in train_l:
    out = std_model(batch)
    assert out.shape == (batch.shape[0], 12, 113), f"Unexpected output shape: {out.shape}"
    out_attn = attn_model(batch)
    assert out_attn.shape == (batch.shape[0], 12, 113)
    break
print("[smoke] forward pass: PASSED")

# Test 5: mini training run (50 epochs)
print("[smoke] Mini training run (50 epochs, hybrid standard)...")
from training.train_v2 import train_v2
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    h = train_v2(
        model=std_model,
        train_loader=train_l, val_loader=val_l, test_loader=test_l,
        num_epochs=50, lr=3e-4, weight_decay=1.0,
        checkpoint_dir=tmpdir, checkpoint_name="smoke_test",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        task_name="hybrid_retrieve_add",
        erank_eval_epochs=[50],
        erank_n_examples=50,
    )
print(f"[smoke] mini training done. final val_acc={h['val_acc'][-1]:.4f}")
print("[smoke] All smoke tests PASSED.")
PYEOF

echo ""
echo "  Smoke test passed."
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Launch main training
# ---------------------------------------------------------------------------

echo "========================================================"
echo "  Phase 2: Main training (18 runs, may take many hours)"
echo "========================================================"
echo "  Seeds:        0, 1, 2"
echo "  Tasks:        key_value, modular_addition, hybrid_retrieve_add"
echo "  Architectures: standard, attention_only"
echo "  Logging to:   ${OUTPUT_DIR}/tmux_run.log (if launched via tmux)"
echo ""
echo "  Progress: partial results saved after each run to"
echo "    ${OUTPUT_DIR}/aggregate/partial_summary.csv"
echo "========================================================"

python scripts/train_thesis_additions.py

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo "  All experiments complete."
echo "  Date: $(date)"
echo "  Results in: ${OUTPUT_DIR}"
echo "  Report:     ${OUTPUT_DIR}/aggregate/thesis_additions_report.md"
echo "========================================================"
