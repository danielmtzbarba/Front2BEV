import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.factory import models
from src.utils import criterions as crit

from dan.utils.torch import load_model

# ----------------------------------------------------------------------------
class Builder(object):

    def __init__(self, config, gpu_id):

        self.model = models.build_model(config)

        if config.distributed:
            self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)
        
        self.model.to(gpu_id)

        self.optimizer = self._build_optimizer(config)
        self.lr_scheduler = self._build_scheduler(config)

        if config.resume:
            self.load_checkpoint()
        
        self.criterion = self._build_criterion(config)
    
    def _build_optimizer(self, config):
        if config.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

        elif config.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), config.learning_rate, 
                        weight_decay=config.weight_decay)
        return optimizer

    def _build_scheduler(self, config):

        # Build learning rate scheduler
        if config.lr_scheduler == 'steplr':
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=config.lr_step_size,
                                                    gamma=config.lr_gamma)
        elif config.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        config.lr_milestones,
                                                            gamma=config.lr_gamma)
        return lr_scheduler

    def _build_criterion(self, config):

        model_name = config.model
        if model_name == 'ved':

            '''
            criterion = crit.VaeOccupancyCriterion(config.prior,
                                            config.xent_weight, 
                                            config.uncert_weight,
                                            config.weight_mode,
                                            config.kld_weight, 
                                            )
            '''
            
            criterion = crit.VedCriterion(config.prior, config.weight_mode,
                                            config.xent_weight, 
                                            config.kld_weight, 
                                            config.uncert_weight,
                                            )
        else:                              

            if config.loss_fn == 'focal':
                criterion = crit.FocalLossCriterion(config.focal.alpha, config.focal.gamma)
            elif config.loss_fn == 'prior':
                criterion = crit.PriorOffsetCriterion(config.prior)
            else:
                criterion = crit.OccupancyCriterion(config.prior, config.xent_weight, 
                                            config.uncert_weight, config.weight_mode)
            
        if config.num_gpus > 0:
            criterion.cuda()
        
        return criterion

    def _load_checkpoint(self, path):
        ckpt = torch.load(path)

        # Load model weights
        if isinstance(self.model, DDP):
            self.model = ckpt.module
        self.model.load_state_dict(ckpt['model'])

        # Load optimiser state
        self.optimizer.load_state_dict(ckpt['optimizer'])

        # Load scheduler state
        self.lr_scheduler.load_state_dict(ckpt['scheduler'])

        return ckpt
    
    def get_train_objs(self):
        return self.model, self.optimizer, self.lr_scheduler, self.criterion
    
    def get_test_objs(self, config):
        ckpt_path = os.path.join(config.logdir, config.name,
                                  config.model, f"{config.name}.pth.tar")
        self._load_checkpoint(ckpt_path)
        return self.model

# ----------------------------------------------------------------------------
