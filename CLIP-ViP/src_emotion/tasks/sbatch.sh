#!/bin/bash

#SBATCH --job-name=clip_64                    # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-5:00:00                     # 1 hour timelimit
#SBATCH --mem=100000MB                         # Using 10GB CPU Memory
#SBATCH --partition=P2                         # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor

eval "$(conda shell.bash hook)"
conda activate EDGE

srun accelerate launch --config_file /home/s2/dayelee/accel.yaml ./run_video_retrieval.py --config /home/s2/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/s2/dayelee --num_train_epochs 5 --output_dir /shared/s2/lab01/dayelee_store/checkpoints/msrvtt --is_train --train_batch_size 16