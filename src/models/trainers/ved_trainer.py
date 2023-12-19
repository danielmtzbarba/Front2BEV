import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.utils.eval import metric_eval_bev


class VedTrainer(nn.Module):
    def __init__(self, model, criterion, config, rank):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.rank = rank
    
    def forward(self, batch, phase):
        images = batch['image'].float().to(self.gpu_id)
        labels = batch['label'].long().to(self.gpu_id)
            
        logits, mu, logvar = self.model(images, phase == 'train')
        loss = self.criterion(logits, labels, mu, logvar)

        return logits, loss
        
    def metrics(self, logits, batch):
        labels = batch['label'].long().numpy().squeeze()
        predictions = np.reshape(np.argmax(logits.cpu().numpy().transpose(
            (0, 2, 3, 1)), axis=3), [64, 64])
        acc, iou = metric_eval_bev(predictions, labels, self.config.num_class)
        return {'acc': acc, 'iou': iou}