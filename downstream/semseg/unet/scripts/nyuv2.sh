#!/bin/bash

python main.py \
    train.batch_size=8 \
    train.phase=train \
    train.workers=4 \
    train.log_dir=$LOG_DIR \
    train.exp_name=NYUv2_SEMSEG \
    train.report_epoch=5 \
    optimizer.optimizer=SGD \
    optimizer.lr=0.01 \
    scheduler.scheduler=PolyLR \
    scheduler.max_epochs=80 \
    dataset.name=NYUv2 \
    dataset.size=[240,320] \
    dataset.resize=False \
    dataset.random_crop=False \
    dataset.path=$DATAPATH \
    distributed.num_gpus=1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.partition=debug \
    hydra.launcher.timeout_min=2000 \
    finetune.model=Semantic2D \
    finetune.backbone=$BACKBONE \
    finetune.pretrain=$INIT \
    
