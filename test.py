# print('hello world')
# srun --account=bfin-delta-gpu --partition=gpuA100x4 --gpus=1 --time=00:30:00 --mem=32g jupyter-notebook --no-browser --port=52068 --ip=0.0.0.0

# srun -A bfin-delta-gpu --time=00:30:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA100x4 --gpus=1 --mem=16g --pty /bin/bash
# salloc --mem=16g --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --partition=ghx4 --account=account_name --time=00:30:00 --gpus-per-node=1

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ambityga/imagenet100")

print("Path to dataset files:", path)


# python dataset_tools.py convert --source=/u/hli46/REPA/imagenet_100/8/train_100_class --dest=/u/hli46/REPA/processed_data_100/images --resolution=256x256 --transform=center-crop-dhariwal
# python dataset_tools.py encode --source=/u/hli46/REPA/processed_data_100/images --dest=/u/hli46/REPA/processed_data_100/vae-sd