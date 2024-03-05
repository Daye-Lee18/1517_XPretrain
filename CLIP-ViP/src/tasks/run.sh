#!/bin/bash 

# 2024-Mar-5 - 1:10 ET 
CUDA_VISIBLE_DEVICES=0,1 taskset -c 0-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/dayelee --num_train_epochs 5 --output_dir /home/dayelee/clipvip/checkpoints/msrvtt

# CUDA_VISIBLE_DEVICES=1 taskset -c 8-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py \
#     --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json \
#     --blob_mount_dir /home/dayelee --num_train_epochs 5 \
#     --output_dir /home/dayelee/clipvip/checkpoints/msrvtt
