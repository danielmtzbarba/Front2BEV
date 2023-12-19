import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.utils.eval import metric_eval_bev


class PonTrainer(nn.Module):
    def __init__(self, model, criterion, config, rank):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config       
        self.rank = rank
        
    def forward(self, batch, phase):

        batch = [t.cuda() for t in batch]
        image, calib, labels, mask = batch

        # Predict class occupancy scores and compute loss
        logits = self.model(image, calib)
        loss = self.criterion(logits, labels, mask)

        return logits, loss
        
    def metrics(self, logits, batch):
        # Update confusion matrix
        scores = logits.cpu().sigmoid()  
        scores > self.config.score_thresh
        acc = 0.0
        iou = 0.0
        return {'acc': acc, 'iou': iou}