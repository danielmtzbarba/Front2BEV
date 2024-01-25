import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.utils.eval import metric_eval_bev

from src.utils.confusion import BinaryConfusionMatrix

class VedTrainer(nn.Module):
    def __init__(self, model, criterion, config, rank):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.rank = rank

        self.reset_metrics()
    
    def reset_metrics(self):
        # Accuracy
        self._acc = 0.0
        # Initialise confusion matrix
        self.cm = BinaryConfusionMatrix(self.config.num_class)

    def forward(self, batch, phase):
        batch = [t.cuda() for t in batch]
        image, calib, labels, mask = batch
        
        logits, mu, logvar = self.model(image, phase == 'train')
        loss = self.criterion(logits, labels, mask, mu, logvar)

        return logits, loss
        
    def metrics(self, logits, batch):

        image, calib, labels, mask = batch

        # Update confusion matrix
        scores = logits.cpu().sigmoid() > self.config.score_thresh
        self.cm.update(scores > self.config.score_thresh, labels, mask)
        return {'acc': self.cm.accuracy, 'iou': self.cm.mean_iou, 'cm': self.cm}
    
