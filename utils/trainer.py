from tqdm import tqdm
import numpy as np
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
import torch.nn.functional as F
import torch

from utils.eval import metric_eval_bev
from dan.utils import save_pkl_file

# ----------------------------------------------------------------------------
def loss_function_map(pred_map, map, mu, logvar, args, rank):
    if args.class_weights is not None:
        args.class_weights = torch.Tensor(args.class_weights).to(rank)

    if args.ignore_class:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=args.class_weights, ignore_index=args.num_class)
    else:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=args.class_weights)

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
        args: object):
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.dataloaders = dataloaders

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch(self, batch_rgb, batch_map_gt):
        self.optimizer.zero_grad()

        # forward: Track history only if training
        with torch.set_grad_enabled(self.phase == 'train'):
            pred_map, mu, logvar = self.model(batch_rgb, self.phase == 'train')

            loss, CE, KLD = loss_function_map(pred_map, batch_map_gt, mu, logvar, self.args, self.gpu_id)

            # backward + optimize only if in training phase
            if self.phase == 'train':
                loss.backward()
                self.optimizer.step()
                
                # Log
                self.log_batch['loss'].append(loss.item())
                self.log_batch['CE_loss'].append(CE.item())
                self.log_batch['KLD_loss'].append(KLD.item())
                
            else:
                # Validation
                bev_gt = batch_map_gt.cpu().numpy().squeeze()
                bev_nn = np.reshape(
                            np.argmax(pred_map.cpu().numpy().transpose(
                                (0, 2, 3, 1)), axis=3), [64, 64])
            
                temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, self.args.num_class)
                self.acc += temp_acc
                self.iou += temp_iou

        self.running_loss += loss.item()

    def _run_epoch(self, epoch):

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.args.batch_size} | Steps: {len(self.dataloaders[self.phase])}")
        self.dataloaders[self.phase].sampler.set_epoch(epoch)

        # Iterate over data.
        for batch in self.dataloaders[self.phase]:
            batch_rgb = batch['rgb'].float().to(self.gpu_id)
            batch_map_gt = batch['map'].long().to(self.gpu_id)

            self._run_batch(batch_rgb, batch_map_gt)

        # ------------------------------
        # Logging per epoch
        # ------------------------------
        if self.phase == 'train':
            self.running_loss = self.running_loss / len(self.dataloaders["train"])
            self.log_epoch['mean_train_loss'].append(self.running_loss)
            print("\nEpoch:", epoch + 1, "Train loss (mean):", self.running_loss, "\n", '-' * 50)

        else:
            self.running_loss = self.running_loss / len(self.dataloaders["val"])
            self.log_epoch['mean_val_loss'].append(self.running_loss)
            print("\nEpoch:", epoch + 1, "Val loss (mean):", self.running_loss, "\n", '-' * 50)

            # ------------------------------
            # Logging metrics and save model
            # ------------------------------

            self.log_epoch['val_acc'].append(self.acc / len(self.dataloaders["val"]))
            print("Val acc: ", self.acc / len(self.dataloaders["val"]))

            self.log_epoch['val_iou'].append(self.iou / len(self.dataloaders["val"])) 
            print("Val mIoU: ", self.iou / len(self.dataloaders["val"]), "\n", '-' * 50)
        
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': ckp,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, self.args.ckpt_path)
        
        print(f"Epoch {epoch} | Training checkpoint saved at {self.args.ckpt_path}")
        print('-' * 50)

    def train(self):

        self.log_batch = {
            'loss': [],
            'CE_loss': [],
            'KLD_loss': [],
        }

        self.log_epoch = {
            'epoch': [],
            'mean_train_loss': [],
            'mean_val_loss': [],
            'val_acc': [],
            'val_iou': [],
        }

        epoch = 0

        while epoch < self.args.n_epochs:

            print('\nEpoch {}/{}'.format(epoch + 1, self.args.n_epochs))
            print('-' * 50)

            self.log_epoch['epoch'].append(epoch)

            for self.phase in ['train', 'val']:

                self.running_loss = 0.0
                self.acc = 0.0
                self.iou = 0.0

                if self.phase == 'train':
                    self.scheduler.step()
                    # Set model to training mode
                    self.model.train()  
                else:
                    # Set model to evaluate mode
                    self.model.eval()  

                self._run_epoch(epoch)

                # ------------------------------
                # Epoch end
                # ------------------------------
                log_dict = {
                    'batches':self.log_batch,
                    'epochs': self.log_epoch
                }    

                save_pkl_file(log_dict, str(self.args.log_path).replace(".pkl", f"gpu_{self.gpu_id}.pkl"))

                if self.gpu_id == 0:
                    self._save_checkpoint(epoch)

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

