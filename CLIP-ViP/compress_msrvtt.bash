#!/usr/bin/env bash

FPS=6
SIZE=224
DATA_DIR="./data/msrvtt"
OUT_DIR="./data/msrvtt/videos_6fps"

python compress.py \
    --input_root=${DATA_DIR} --output_root=${OUT_DIR} \
    --fps=${FPS} --size=${SIZE} --file_type=video --num_workers 24