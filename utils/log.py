import os
from dan.utils import save_pkl_file

class TrainLog(object):

    def __init__(self, config):
        
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
    
    def new_epoch(self, epoch, gpu_id, steps):
        if gpu_id == 0:
            print('-' * 50,'\nEpoch {}/{}'.format(epoch, self.config.num_epochs))
        
        print( '-' * 50,f"\n[GPU{gpu_id}] Epoch {epoch} | Batchsize: {self.config.batch_size} | Steps: {steps}", "\n", '-' * 50)
        self._epochs['epoch'].append(epoch)
        

    def log_batch(self, loss):
        self._batches['loss'].append(loss)

    def log_epoch(self, epoch, running_loss, phase):
        self._epochs[f'mean_{phase}_loss'].append(running_loss)
        print('-' * 50, "\nEpoch:", epoch, f"{phase} loss (mean):", running_loss)
    
    def log_metrics(self, cm, acc, iou):
        self._epochs['val_iou'].append(iou)
        print('-' * 50, "\nVal IoU: ", iou)

        self._epochs['val_acc'].append(acc) 
        print("Val acc: ", acc)

    def save_log(self):
        log_dict = {
            'batches':self._batches,
            'epochs': self._epochs
        }  

        log_path = os.path.join(self.config.logdir, f'{self.config.name}.pkl')

        save_pkl_file(log_dict, log_path)