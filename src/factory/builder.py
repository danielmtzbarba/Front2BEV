import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.factory import models
from src.utils import criterions as crit
from src.models import trainers
# ----------------------------------------------------------------------------
class Builder(object):

    def __init__(self, config, gpu_id):

        self._gpu_id = gpu_id
        self._config = config

        self.model = models.build_model(config)

        self.model.to(gpu_id)
        
        if config.distributed:
            self.model = DDP(self.model, device_ids=[gpu_id],
                             find_unused_parameters=config.find_unused_params)

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()

        if config.resume:
            self._load_checkpoint(f'{config.logdir}/{config.name}/{config.model}/{config.name}.pth.tar')
            print(f"{config.model} model loaded! Resuming training...")
        
        self.criterion = self._build_criterion()
    
    def _build_optimizer(self):
        if self._config.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self._config.learning_rate,
                                    weight_decay=self._config.weight_decay)

        elif self._config.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), self._config.learning_rate, 
                                    weight_decay=self._config.weight_decay)
        return optimizer

    def _build_scheduler(self):

        # Build learning rate scheduler
        if self._config.lr_scheduler == 'steplr':
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=self._config.lr_step_size,
                                                    gamma=self._config.lr_gamma)
        elif self._config.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        self._config.lr_milestones,
                                                            gamma=self._config.lr_gamma)
        return lr_scheduler

    def _build_criterion(self):
        model_name = self._config.model

        if model_name == 'ved':
            if self._config.weight_mode == 'recall':
                criterion = crit.VaeRecallCriterion(self._config.prior,
                                                    self._config.xent_weight, 
                                                    self._config.uncert_weight,
                                                    self._config.kld_weight,
                                                    self._config.num_class, 
                                                       )
            else:
                criterion = crit.VaeOccupancyCriterion(self._config.prior,
                                                       self._config.xent_weight, 
                                                       self._config.uncert_weight,
                                                       self._config.weight_mode,
                                                       self._config.kld_weight, 
                                                       )
        else:
                
            if self._config.loss_fn == 'focal':
                criterion = crit.FocalLossCriterion(self._config.focal.alpha, self._config.focal.gamma)

            elif self._config.loss_fn == 'prior':
                criterion = crit.PriorOffsetCriterion(self._config.prior)
           
            elif self._config.weight_mode == 'recall':
                criterion = crit.RecallCriterion(self._config.prior,  self._config.xent_weight, 
                                                    self._config.uncert_weight, self._config.num_class) 
            else:
                criterion = crit.OccupancyCriterion(self._config.prior, self._config.xent_weight, 
                                                    self._config.uncert_weight, self._config.weight_mode)
            
        if self._config.num_gpus > 0:
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
    
# ----------------------------------------------------------------------------

    def attach2trainer(self):
        if self._config.model == "ved":
            trainer = trainers.VedTrainer(self.model, self.criterion, 
                                self._config, self._gpu_id)
            
        if self._config.model == "pon":
            trainer = trainers.PonTrainer(self.model, self.criterion,
                                self._config, self._gpu_id)
        
        return trainer
        
    def get_train_objs(self):
        return self.attach2trainer(), self.optimizer, self.lr_scheduler
    
    def get_test_objs(self):
        ckpt_path = os.path.join(self._config.logdir, self._config.name,
                                  self._config.model, f"{self._config.name}.pth.tar")
        _  = self._load_checkpoint(ckpt_path)
        print("Loaded model at", ckpt_path)
        return self.attach2trainer()

# ----------------------------------------------------------------------------
