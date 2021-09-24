#!/bin/bash
python main.py \
    train.batch_size=2 \
    train.phase=train \
    train.report_epoch=1 \
	train.workers=4 \
    train.log_dir=$LOG_DIR \
    train.exp_name=Pri3D_MegaDepth \
    scheduler.scheduler=PolyLR \
    scheduler.max_epochs=5 \
    optimizer.optimizer=SGD \
    optimizer.lr=0.1 \
	optimizer.accumulate_step=1 \
    dataset.name=MegaDepth \
	dataset.size=[240,320] \
	dataset.path=$DATAPATH \
    pretrain.pretrained=imagenet \
    pretrain.model=Pri3D \
    pretrain.nceT=0.4 \
    pretrain.backbone=$BACKBONE \
	pretrain.thresh=0.05 \
	pretrain.sample_points=1024 \
    pretrain.view_invariant=True \
    pretrain.geometric_prior=False \
    pretrain.depth=True \
    distributed.num_gpus=1 \
