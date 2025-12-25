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


# Inception Score: 18.110876083374023
# FID: 89.82761825760417
# sFID: 21.352788509765105
# Precision: 0.2226
# Recall: 0.6141

# Inception Score: 1.9074647426605225
# FID: 255.52646042261287
# sFID: 127.39290336197519
# Precision: 0.28736
# Recall: 0.681