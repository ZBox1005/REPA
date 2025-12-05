#!/bin/bash

set -e

REF_NPZ="evaluations/imagenet100_reference.npz"
SAMPLE_NPZ="/u/hli46/REPA/samples/SiT-B-2-0050000-size-256-vae-ema-cfg-1.8-seed-0-sde.npz"


echo "=== ImageNet-100 Evaluation Pipeline ==="
echo ""

python evaluations/evaluator.py \
    "$REF_NPZ" \
    "$SAMPLE_NPZ"

echo ""
echo "=== Evaluation Complete ==="