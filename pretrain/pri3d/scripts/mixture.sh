#!/bin/bash
python main.py \
    train.batch_size=64 \
    train.phase=train \
    train.report_epoch=1 \
	train.workers=4 \
    train.log_dir=$LOG_DIR \
    train.exp_name=Pri3D \
    scheduler.scheduler=PolyLR \
    scheduler.max_epochs=5 \
    optimizer.optimizer=SGD \
    optimizer.lr=0.1 \
	optimizer.accumulate_step=1 \
    dataset.name=ScanNet \
	dataset.size=[240,320] \
    pretrain.pretrained=imagenet \
    pretrain.model=Pri3D \
    pretrain.nceT=0.4 \
    pretrain.backbone=$BACKBONE \
	pretrain.thresh=0.05 \
	pretrain.sample_points=1024 \
    pretrain.view_invariant=$VIEW \
    pretrain.geometric_prior=$GEO \
    pretrain.depth=True \
    distributed.num_gpus=8 \
