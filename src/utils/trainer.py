from tqdm import tqdm
import numpy as np
import os

import torch.distributed as dist 
import torch

from src.utils.logger import TrainLog

class Trainer:
    def __init__(
        self,
        dataloaders: dict,
        model_trainer: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        config: object):
        
        self.gpu_id = gpu_id
        self._model_trainer = model_trainer.to(gpu_id)
        self.dataloaders = dataloaders

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.train_log = TrainLog(self.config)
        self.log = True if gpu_id == 0 else False

    def _backward_plus_optimize(self, loss):
         # backward + optimize only if in training phase
        loss.backward()
        self.optimizer.step()
        
    def _calc_metrics(self, logits, batch):
        metrics = self._model_trainer.metrics(logits, batch)

        self.acc += metrics['acc'] / len(self.dataloaders["val"])
        self.iou += metrics['iou'] / len(self.dataloaders["val"])
        self.cm = metrics['cm']

    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        # forward: Track history only if training
        with torch.set_grad_enabled(self._phase == 'train'):

            # forward
            logits, loss = self._model_trainer(batch, self._phase)

            if self._phase == "train":
                # Training
                if self.log:
                    self.train_log.log_batch(loss.item(), self._iteration)

                    if self._iteration % self.config.log_interval == 0:
                        self.train_log.log_visual_res(batch, logits, self._iteration, self._phase)
                
                self._backward_plus_optimize(loss)
               
            else:
                # Validation
                self._calc_metrics(logits, batch)

                if self.log and (self._iteration % self.config.log_interval == 0):
                    self.train_log.log_visual_res(batch, logits, self._iteration, self._phase)

        self.running_loss += loss.item()
        self._iteration += 1
        
    def _run_epoch(self):

        if self.config.distributed:
            self.dataloaders[self._phase].sampler.set_epoch(self._epoch)

        # Iterate over data
        for batch in tqdm(self.dataloaders[self._phase], disable=(self.gpu_id != 0)):
            self._run_batch(batch)

        self.running_loss = self.running_loss / len(self.dataloaders[self._phase])


    def _run_train_iter(self):
        # Reset variables
        self.running_loss = 0.0
        self.scheduler.step()
        #
        self._iteration = (self._epoch - 1) * len(self.dataloaders["train"]) + 1 

        # Set model to training mode
        self._model_trainer.model.train()  
        self._run_epoch()

        # Logging epoch loss
        if self.log:
            self.train_log.log_epoch(self._epoch, self.running_loss, self._phase)

    def _run_val_iter(self):
        # Reset variables
        self._model_trainer.reset_metrics()
        self._iteration = 1 
        self.running_loss = 0.0
        #
        self.acc = 0.0
        self.iou = 0.0
        self.cm = None
        
        # Set model to eval mode
        self._model_trainer.model.eval()  
        self._run_epoch()

        if self.log:
            # Logging epoch loss
            self.train_log.log_epoch(self._epoch, self.running_loss, self._phase)
            # Logging metrics
            self.train_log.log_metrics(self.acc, self.iou, self.cm, self._epoch)

        return self.iou
    
    def _save_checkpoint(self):

        if self.config.distributed:
            model_ckpt = self._model_trainer.model.module
        else:
            model_ckpt = self._model_trainer.model

        logdir = os.path.join(os.path.expandvars(self.config.logdir), self.config.name, self.config.model)
        ckpt_path = os.path.join(logdir, f'{self.config.name}.pth.tar')

        ckpt = {
            'model' : model_ckpt.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
            'epoch' : self._epoch,
            'best_iou' : self._best_iou
        }

        torch.save(ckpt, ckpt_path)

        print('-' * 50, f"\nEpoch {self._epoch} | Training checkpoint saved at {ckpt_path}")
        print('-' * 50, "\n")

# -----------------------------------------------------------------------------
    
    def train(self):

        self._best_iou = 0.0

        self._epoch = 1
        self._iteration = 1

        # Traini loop

        while self._epoch <= self.config.num_epochs:

            self._phase = 'train'
            if self.log:
                self.train_log.log_phase(self._epoch, self.gpu_id,
                                        len(self.dataloaders[self._phase]), self._phase)
            self._run_train_iter()        

            self._phase = 'val'
            if self.log:
                self.train_log.log_phase(self._epoch, self.gpu_id,
                                        len(self.dataloaders[self._phase]), self._phase)            
            iou = self._run_val_iter()
    
            # Epoch end
            if self.log:
                self.train_log.save_log()

                if iou > self._best_iou:
                    self._save_checkpoint()
                    self._best_iou = iou

            self._epoch += 1

        # ------------------------------
        # Training end
        # ------------------------------
        print('-' * 50, f"\nTraining ended successfully")
        print('-' * 50)
        # ------------------------------

# -----------------------------------------------------------------------------

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
# -----------------------------------------------------------------------------
