#!/bin/bash
# Train all 4 models in parallel, one per GPU.

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/train_all.py --run 0 2>&1 | tee logs/run0.log &
CUDA_VISIBLE_DEVICES=1 python scripts/train_all.py --run 1 2>&1 | tee logs/run1.log &
CUDA_VISIBLE_DEVICES=4 python scripts/train_all.py --run 2 2>&1 | tee logs/run2.log &
CUDA_VISIBLE_DEVICES=5 python scripts/train_all.py --run 3 2>&1 | tee logs/run3.log &

echo "All 4 training runs launched. Monitor with:"
echo "  grep 'Epoch\|Early stopping\|Final\|Run:\|Device' logs/run*.log"

wait
echo "All runs complete."
