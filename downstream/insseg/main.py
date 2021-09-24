from torch.multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)


import os
import sys 
import random
import json

from detectron2.engine import default_argument_parser, launch, default_setup, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.WEIGHTS == 'imagenet':
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main_func(args):
    cfg = setup(args)

    # register data
    if 'scannet' in cfg.DATASETS.TRAIN[0]:
        register_coco_instances(cfg.DATASETS.TRAIN[0], {}, os.path.join(cfg.DATASETS.ANNOTATION_PATH, "{}.coco.json".format(cfg.DATASETS.TRAIN[0])), cfg.DATASETS.IMAGE_PATH)
        register_coco_instances("scannet_val", {}, os.path.join(cfg.DATASETS.ANNOTATION_PATH, "scannet_val.coco.json"), cfg.DATASETS.IMAGE_PATH)

    if 'nyu' in cfg.DATASETS.TRAIN[0]:
        register_coco_instances("nyu_train", {}, os.path.join(cfg.DATASETS.ANNOTATION_PATH, "nyu_train.coco.json", cfg.DATASETS.IMAGE_PATH))
        register_coco_instances("nyu_val", {}, os.path.join(cfg.DATASETS.ANNOTATION_PATH, "nyu_val.coco.json", cfg.DATASETS.IMAGE_PATH))

    from model.model import MyTrainer
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main_func,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
