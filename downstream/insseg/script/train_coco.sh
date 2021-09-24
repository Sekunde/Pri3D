#!/bin/bash
#SBATCH --gpus=8 \
#SBATCH --cpus-per-gpu=10 \
#SBATCH --mem=64gb \
#SBATCH --time=24:00:00 \

mkdir -p $LOG_DIR

python main.py --num-gpus 8 --config-file config/mask_rcnn_R_50_FPN_COCO.yaml MODEL.WEIGHTS $INIT OUTPUT_DIR $LOG_DIR
