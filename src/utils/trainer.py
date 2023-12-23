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
        self._model_trainer = model_trainer
        self.dataloaders = dataloaders

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.train_log = TrainLog(self.config)
        self.log = True if gpu_id == 0 else False

    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        # forward: Track history only if training
        with torch.set_grad_enabled(self.phase == 'train'):

            # forward
            logits, loss = self._model_trainer(batch, self.phase)

            # backward + optimize only if in training phase
            if self.phase == 'train':
                loss.backward()

                # Log
                if self.log:
                    self.train_log.log_batch(loss.item(), self._iteration)
                
                self.optimizer.step()
                self._iteration += 1
                
            else:
                # Validation
                metrics = self._model_trainer.metrics(logits, batch)

                self.acc += metrics['acc'] / len(self.dataloaders["val"])
                self.iou += metrics['iou'] / len(self.dataloaders["val"])
                self.cm = metrics['cm']

                if self.log:
                    self.train_log.log_visual_res(batch, logits, self._iteration, self.phase)

        self.running_loss += loss.item()

    def _run_epoch(self, epoch):

        if self.config.distributed:
            self.dataloaders[self.phase].sampler.set_epoch(epoch)

        # Iterate over data
        for batch in tqdm(self.dataloaders[self.phase], disable=(self.gpu_id != 0)):
            self._run_batch(batch)

        self.running_loss = self.running_loss / len(self.dataloaders[self.phase])


    def _run_train_iter(self, epoch):
        # Reset variables
        self.running_loss = 0.0
        self.scheduler.step()

        # Set model to training mode
        self._model_trainer.model.train()  
        self._run_epoch(epoch)

        # Logging epoch loss
        if self.log:
            self.train_log.log_epoch(epoch, self.running_loss, self.phase)

    def _run_val_iter(self, epoch):
        # Reset variables
        self._model_trainer.reset_metrics()
        self.running_loss = 0.0
        self.acc = 0.0
        self.iou = 0.0
        self.cm = None
    
        # Set model to eval mode
        self._model_trainer.model.eval()  
        self._run_epoch(epoch)

        if self.log:
            # Logging epoch loss
            self.train_log.log_epoch(epoch, self.running_loss, self.phase)
            # Logging metrics
            self.train_log.log_metrics(self.acc, self.iou, self.cm, epoch)

        return self.iou
    
    def _save_checkpoint(self, epoch, best_iou):

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
            'epoch' : epoch,
            'best_iou' : best_iou
        }

        torch.save(ckpt, ckpt_path)

        print('-' * 50, f"\nEpoch {epoch} | Training checkpoint saved at {ckpt_path}")

# -----------------------------------------------------------------------------
    
    def train(self):

        epoch = 1
        self.best_iou = 0.0

        self._iteration = (epoch - 1) * len(self.dataloaders["train"])

        while epoch <= self.config.num_epochs:
            self.phase = 'train'
            if self.log:
                self.train_log.log_phase(epoch, self.gpu_id,
                                        len(self.dataloaders[self.phase]), self.phase)
            self._run_train_iter(epoch)        

            self.phase = 'val'
            if self.log:
                self.train_log.log_phase(epoch, self.gpu_id,
                                        len(self.dataloaders[self.phase]), self.phase)            
            iou = self._run_val_iter(epoch)
    
            # Epoch end
            if self.log:
                self.train_log.save_log()

                if iou > self.best_iou:
                    self._save_checkpoint(epoch, iou)
                    self.best_iou = iou

            epoch += 1

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