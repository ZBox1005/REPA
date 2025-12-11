export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

accelerate launch --num_processes=1 --num_machines=1 train_vanilla.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --resume-step=30000 \
  --max-train-steps=50000 \
  --encoder-depth=8 \
  --num-workers 8 \
  --output-dir="exps" \
  --exp-name="sit-B-exp2" \
  --checkpointing-steps=10000 \
  --sampling-steps=5000 \
  --data-dir=/u/hli46/REPA/processed_data_100/