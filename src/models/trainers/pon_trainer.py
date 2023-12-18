import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.utils.eval import metric_eval_bev


class PonTrainer(nn.Module):
    def __init__(self, model, criterion, num_class, rank):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_class = num_class
        self.rank = rank
    
    def forward(self, images, labels, phase):
        logits = self.model(images)
        loss = self.criterion(logits, labels)

        return logits, loss
        
    def metrics(self, logits, labels):
        labels = labels.cpu().numpy().squeeze()
        predictions = np.reshape(np.argmax(logits.cpu().numpy().transpose(
            (0, 2, 3, 1)), axis=3), [64, 64])
    
        return metric_eval_bev(predictions, labels, self.num_class)