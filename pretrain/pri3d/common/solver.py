import logging

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
from torch import nn



def initialize_optimizer(params, config):
  if config.optimizer.optimizer == 'SGD':
    return SGD(params, lr=config.optimizer.lr,
               momentum=config.optimizer.sgd_momentum,
               dampening=config.optimizer.sgd_dampening,
               weight_decay=config.optimizer.weight_decay)
  elif config.optimizer.optimizer == 'Adam':
    return Adam(params, lr=config.optimizer.lr,
                betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
                weight_decay=config.optimizer.weight_decay)
  else:
    logging.error('Optimizer type not supported')
    raise ValueError('Optimizer type not supported')

class PolyLR(LambdaLR):
  """DeepLab learning rate policy"""
  def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_epoch)


def initialize_scheduler(optimizer, config, epoch_size, last_epoch=-1):
  if config.scheduler.scheduler == 'StepLR':
    return StepLR(optimizer, 
                  step_size=config.scheduler.decay_epochs*epoch_size, 
                  gamma=config.scheduler.lr_decay, 
                  last_epoch=last_epoch)
  elif config.scheduler.scheduler == 'MultiStepLR':
    return MultiStepLR(optimizer, 
                       milestones=[epoch*epoch_size for epoch in config.scheduler.decay_epochs], 
                       gamma=config.scheduler.lr_decay,
                       last_epoch=last_epoch)
  elif config.scheduler.scheduler == 'PolyLR':
    return PolyLR(optimizer, 
                  max_iter=config.scheduler.max_epochs*epoch_size, 
                  power=config.scheduler.poly_power, 
                  last_epoch=last_epoch)
  else:
    logging.error('Scheduler not supported')

def set_bn_momentum_default(bn_momentum):
    #TODO Debug if .parameters() needed for model
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


def initialize_bnm_scheduler(model):
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    BN_DECAY_STEP = 20
    BN_DECAY_RATE = 0.5
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    return BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=-1)
