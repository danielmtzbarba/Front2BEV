import os
from dan.utils import save_pkl_file

from torch.utils.tensorboard import SummaryWriter

from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES

from src.utils.visualize import colorise

class TrainLog(object):

    def __init__(self, config):
        # Create tensorboard summary 
        self._summary = SummaryWriter(os.path.join(config.logdir, config.name, config.model))
        
        self.config = config
        self._batches = {
            'loss': [],
        }

        self._epochs = {
            'epoch': [],
            'mean_train_loss': [],
            'mean_val_loss': [],
            'val_acc': [],
            'val_iou': [],
            'val_miou': [],
        }
    
    def log_phase(self, epoch, gpu_id, steps, phase="train"):
        print('-' * 50,'\nEpoch {}/{}'.format(epoch, self.config.num_epochs))
        print('-' * 50, f"\n{phase.capitalize()}")
        print('-' * 50, f"\n[GPU{gpu_id}] Epoch {epoch} | Batchsize: {self.config.batch_size} | Steps: {steps}")
        print('-' * 50)
        
        if phase == "train":
            self._epochs['epoch'].append(epoch)
        
    def log_batch(self, loss, iteration):
        # Update tensorboard
        self._summary.add_scalar('train/loss', float(loss), iteration)

        # Log
        self._batches['loss'].append(loss)

    def log_epoch(self, epoch, running_loss, phase):
        # Update tensorboard
        self._summary.add_scalar(f'{phase}/mloss', float(running_loss), epoch)
        
        # Log
        self._epochs[f'mean_{phase}_loss'].append(running_loss)
        print("Epoch:", epoch, f"{phase} loss (mean):", running_loss)
        
    def log_metrics(self, acc, iou, confusion, epoch):
        class_names = NUSCENES_CLASS_NAMES 

        self._summary.add_scalar(f'val/miou', confusion.mean_iou, epoch)

        print('-' * 50, '\nResults:')
        for name, iou_score in zip(class_names, confusion.iou):
            print('{:20s} {:.3f}'.format(name, iou_score)) 

        self._epochs['val_iou'].append(iou)
        print("\nVal mIoU: ", confusion.mean_iou)

        self._epochs['val_acc'].append(acc) 
        print("Val acc: ", acc)
    
    def log_visual_res(self, batch, logits, iteration, phase):

        image, calib, labels, mask = batch
        scores = logits.cpu().sigmoid() > self.config.score_thresh

        self._summary.add_image(phase + '/image', image[0], iteration, dataformats='CHW')
        self._summary.add_image(phase + '/pred', colorise(scores[0], 'coolwarm', 0, 1),
                        iteration, dataformats='NHWC')
        self._summary.add_image(phase + '/gt', colorise(labels[0], 'coolwarm', 0, 1),
                        iteration, dataformats='NHWC')
    
 
    def save_log(self):
        log_dict = {
            'batches':self._batches,
            'epochs': self._epochs
        }  

        log_path = os.path.join(self.config.logdir, self.config.name,
                                 self.config.model,f'{self.config.name}.pkl')

        save_pkl_file(log_dict, log_path)