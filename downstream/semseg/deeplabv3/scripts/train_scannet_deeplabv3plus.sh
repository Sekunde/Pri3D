#!/bin/bash

python main.py \
    train.log_dir=$LOG_DIR \
    train.exp_name=DeepLabV3Plus_SEMSEG \
    train.model=deeplabv3plus \
    train.weight=$INIT \
    dataset.data_root=$DATAPATH \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.partition=debug \
    hydra.launcher.timeout_min=2000 \
