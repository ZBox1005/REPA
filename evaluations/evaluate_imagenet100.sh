#!/bin/bash

set -e

REF_NPZ="evaluations/imagenet100_reference.npz"
SAMPLE_NPZ="/path/to/your/samples.npz"


echo "=== ImageNet-100 Evaluation Pipeline ==="
echo ""

python evaluations/evaluator.py \
    "$REF_NPZ" \
    "$SAMPLE_NPZ"

echo ""
echo "=== Evaluation Complete ==="