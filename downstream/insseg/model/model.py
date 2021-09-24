import os
from detectron2.engine import DefaultTrainer, default_argument_parser
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        if 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif 'scannet' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif 'voc' in dataset_name:
            return PascalVOCDetectionEvaluator(dataset_name)
        else:
            raise NotImplementedError