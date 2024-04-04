#!/usr/bin/env bash

# FIRST ARGUMENT: DATA_DIR
DATA_DIR=$1
# SAVE PATH
OUT_DIR="${DATA_DIR}/msrvtt/videos_6fps"

FPS=6
SIZE=224

python compress.py \
    --input_root=${DATA_DIR} --output_root=${OUT_DIR} \
    --fps=${FPS} --size=${SIZE} --file_type=video --num_workers 24