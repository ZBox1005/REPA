# Evaluation Tools for ImageNet-100

This directory contains evaluation scripts for computing image generation metrics (FID, IS, sFID, Precision, Recall) on ImageNet-100 dataset.

## Prerequisites

### 1. Download Reference Statistics

Download the pre-computed ImageNet-100 reference statistics file:

**Download Link**: [imagenet100_reference.npz](https://drive.google.com/file/d/1eFxddOoHsY2leQWFKWSLwTrCHOo_qZaa/view?usp=drive_link)

Place the downloaded file in the `evaluations/` directory.


Your directory structure should look like:
```
evaluations/
├── README.md
├── imagenet100_reference.npz    # Downloaded file
├── evaluator.py
├── requirements.txt
└── evaluate_imagenet100.sh
```

## Usage

### Quick Start

Evaluate your generated samples using the provided script:

```bash
cd /path/to/REPA

bash evaluations/evaluate_imagenet100.sh
```

### Manual Evaluation

If you want to run evaluation manually:

```bash
cd /path/to/REPA

python evaluations/evaluator.py \
    evaluations/imagenet100_reference.npz \
    /path/to/your/samples.npz
```

## My Evaluation Results for Reference

### SiT-B/2

#### Configuration
```bash
"--report-to=wandb",
"--allow-tf32",
"--mixed-precision=fp16",
"--seed=0",
"--path-type=linear",
"--prediction=v",
"--weighting=uniform",
"--model=SiT-B/2",
"--num-classes=100",
"--enc-type=dinov2-vit-b",
"--proj-coeff=0.5",
"--encoder-depth=8",
"--output-dir=exps",
"--exp-name=in100-sit-b-quick-v2",
"--data-dir=/path/to/your/imagenet-100",
"--max-train-steps=50000",
"--checkpointing-steps=5000",
"--sampling-steps=2500"
```
#### Results
```bash
Inception Score: 36.54213333129883
FID: 37.606172606055736
sFID: 42.17939442935517
Precision: 0.3878
Recall: 0.7233
```

### SiT-XL/2

#### Configuration
```bash
"--report-to=wandb",
"--allow-tf32",
"--mixed-precision=fp16",
"--seed=0",
"--path-type=linear",
"--prediction=v",
"--weighting=uniform",
"--model=SiT-XL/2",
"--num-classes=100",
"--enc-type=dinov2-vit-b",
"--proj-coeff=0.5",
"--encoder-depth=8",
"--output-dir=exps",
"--exp-name=in100-sit-xl-standard-v2",
"--data-dir=/path/to/your/imagenet-100",
"--batch-size=256",
"--learning-rate=1e-4",
"--max-train-steps=100000",
"--checkpointing-steps=10000",
"--sampling-steps=5000"
```
#### Results
```bash
Inception Score: 59.3747444152832
FID: 9.190527739278878
sFID: 40.55530663121624
Precision: 0.628
Recall: 0.7034
```
