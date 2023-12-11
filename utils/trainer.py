from tqdm import tqdm
import numpy as np
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
import torch.nn.functional as F
import torch

from utils.eval import metric_eval_bev
from dan.utils import save_pkl_file
from utils.log import TrainLog

# ----------------------------------------------------------------------------
def loss_function_map(pred_map, map, mu, logvar, config, rank):
    if config.class_weights is not None:
        config.class_weights = torch.Tensor(config.class_weights).to(rank)

    if config.ignore_class:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=config.class_weights, ignore_index=config.num_class)
    else:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=config.class_weights)

    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.9*CE + 0.1*KLD, CE, KLD
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        dataloaders: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        config: object):
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.dataloaders = dataloaders

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.train_log = TrainLog(self.config)
        self.log = True if gpu_id == 0 else False

    def _run_batch(self, batch_rgb, batch_map_gt):
        self.optimizer.zero_grad()

        # forward: Track history only if training
        with torch.set_grad_enabled(self.phase == 'train'):
            pred_map, mu, logvar = self.model(batch_rgb, self.phase == 'train')

            loss, CE, KLD = loss_function_map(pred_map, batch_map_gt, mu, logvar, self.config, self.gpu_id)

            # backward + optimize only if in training phase
            if self.phase == 'train':
                loss.backward()
                self.optimizer.step()
                
                # Log
                if self.log:
                    self.train_log.log_batch(loss.item())
                
            else:
                # Validation
                bev_gt = batch_map_gt.cpu().numpy().squeeze()
                bev_nn = np.reshape(
                            np.argmax(pred_map.cpu().numpy().transpose(
                                (0, 2, 3, 1)), axis=3), [64, 64])
            
                temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, self.config.num_class)
                self.acc += temp_acc / len(self.dataloaders["val"])
                self.iou += temp_iou / len(self.dataloaders["val"])

        self.running_loss += loss.item()

    def _run_epoch(self, epoch):

        if self.config.distributed:
            self.dataloaders[self.phase].sampler.set_epoch(epoch)

        # Iterate over data
        for batch in tqdm(self.dataloaders[self.phase], disable=(self.gpu_id != 0)):
            batch_rgb = batch['rgb'].float().to(self.gpu_id)
            batch_map_gt = batch['map'].long().to(self.gpu_id)

            self._run_batch(batch_rgb, batch_map_gt)

        if self.phase == 'train':
            self.running_loss = self.running_loss / len(self.dataloaders["train"])

        else:
            self.running_loss = self.running_loss / len(self.dataloaders["val"])

    def _run_train_iter(self, epoch):
        self.phase = 'train'

        # Reset variables
        self.running_loss = 0.0
        self.scheduler.step()

        # Set model to training mode
        self.model.train()  
        self._run_epoch(epoch)

        # Logging epoch loss
        if self.log:
            self.train_log.log_epoch(epoch, self.running_loss, self.phase)

    def _run_val_iter(self, epoch):
        self.phase = 'val'
        # Reset variables
        self.confusion_m = None #BinaryConfusionMatrix(self.config.num_class)
        self.running_loss = 0.0
        self.acc = 0.0
        self.iou = 0.0
    
        # Set model to eval mode
        self.model.eval()  
        self._run_epoch(epoch)

        if self.log:
            # Logging epoch loss
            self.train_log.log_epoch(epoch, self.running_loss, self.phase)
            # Logging metrics
            self.train_log.log_metrics(self.confusion_m, self.acc, self.iou)

        return self.iou #self.confusion_m.mean_iou
    
    def _save_checkpoint(self, epoch, best_iou):

        if self.config.distributed:
            model_ckpt = self.model.module
        else:
            model_ckpt = self.model

        ckpt_path = os.path.join(self.config.logdir, f'{self.config.name}.pth.tar')

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
        self.phase = 'train'

        while epoch <= self.config.num_epochs:
            if self.log:
                self.train_log.new_epoch(epoch, self.gpu_id,
                                        len(self.dataloaders[self.phase]))

            self._run_train_iter(epoch)
            iou = self._run_val_iter(epoch)
    
            # Epoch end
            if self.log:
                self.train_log.save_log()

                #if iou > self.best_iou:
                if True:
                    self._save_checkpoint(epoch, iou)
                    self.best_iou = iou

            epoch += 1

        # ------------------------------
        # Training end
        # ------------------------------
        print('\nTraining ended')
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