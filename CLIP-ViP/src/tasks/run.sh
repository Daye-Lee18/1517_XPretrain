#!/bin/bash 

# 2024-Mar-5 - 1:10 ET 
# CUDA_VISIBLE_DEVICES=0,1 taskset -c 0-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/dayelee --num_train_epochs 5 --output_dir /home/dayelee/clipvip/checkpoints/msrvtt --is_emotion true

# CUDA_VISIBLE_DEVICES=1 taskset -c 8-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py \
#     --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json \
#     --blob_mount_dir /home/dayelee --num_train_epochs 5 \
#     --output_dir /home/dayelee/clipvip/checkpoints/msrvtt

# 2024-Mar-6 - 2:50 ET = finetuning with training 9k and val with 1k test dataset 
# CUDA_VISIBLE_DEVICES=5,6 taskset -c 40-55 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/dayelee --num_train_epochs 5 --output_dir /home/dayelee/clipvip/checkpoints/msrvtt --is_train
# bs 16 
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/dayelee --num_train_epochs 5 --output_dir /home/dayelee/clipvip/checkpoints/msrvtt --is_train --train_batch_size 16


# GSDS server 
# accelerate launch --config_file /home/s2/dayelee/accel.yaml ./run_video_retrieval.py --config /home/s2/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/s2/dayelee --num_train_epochs 5 --output_dir /shared/s2/lab01/dayelee_store/checkpoints/msrvtt --is_train --train_batch_size 16

# 2024-Mar 8 12:45 ET - 현빈 test method test 
accelerate launch --config_file /home/s2/dayelee/accel.yaml ./run_video_retrieval_val.py --config /home/s2/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/s2/dayelee --num_train_epochs 5 --output_dir /shared/s2/lab01/dayelee_store/checkpoints/msrvtt --train_batch_size 8

# bs 32 
# CUDA_VISIBLE_DEVICES=0,1 taskset -c 0-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/s2/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/s2/dayelee --num_train_epochs 5 --output_dir /shared/s2/lab01/dayelee_store/checkpoints/msrvtt --is_train --train_batch_size 32

# test with emotion (without --is_train)
# CUDA_VISIBLE_DEVICES=0,1 taskset -c 0-15 accelerate launch --config_file /home/dayelee/accel.yaml ./run_video_retrieval.py --config /home/dayelee/1517_XPretrain/CLIP-ViP/src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json --blob_mount_dir /home/dayelee --num_train_epochs 5 --output_dir /home/dayelee/clipvip/checkpoints/msrvtt --is_train

# test with neutral (without --is_train)
