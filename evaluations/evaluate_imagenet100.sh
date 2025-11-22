#!/bin/bash

set -e

DATA_PATH="/common/users/bz362/dataset/imagenet-100/images-256" 
REF_NPZ="/common/users/bz362/REPA/evaluations/imagenet100_reference.npz"
SAMPLE_NPZ="/common/users/bz362/REPA/samples/SiT-B-2-final-size-256-vae-ema-cfg-1.8-seed-0-sde.npz"


echo "=== ImageNet-100 Evaluation Pipeline ==="
echo "Data path: $DATA_PATH"
echo ""

python evaluations/evaluator.py \
    "$REF_NPZ" \
    "$SAMPLE_NPZ"

echo ""
echo "=== Evaluation Complete ==="