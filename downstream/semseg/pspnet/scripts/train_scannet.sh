#!/bin/bash

python main.py \
    train.log_dir=$LOG_DIR \
    train.exp_name=PSPNet_SEMSEG \
    train.weight=$INIT \
    dataset.data_root=$DATAPATH \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.partition=debug \
    hydra.launcher.timeout_min=2000 \
