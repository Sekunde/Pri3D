#!/usr/bin/env python3
import hydra
import os 
from common.distributed import multi_proc_run

def single_proc_run(config):
  from common.pretrain import Pretrain
  #dir_path = os.path.dirname(os.path.realpath(__file__))
  trainer = Pretrain(config)
  trainer.train()

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
  if config.distributed.num_gpus > 1:
      multi_proc_run(config.distributed.num_gpus, fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)

if __name__ == '__main__':
  #os.environ["CUDA_VISIBLE_DEVICES"]="5"
  os.environ["OMP_NUM_THREADS"]="12"
  main()
