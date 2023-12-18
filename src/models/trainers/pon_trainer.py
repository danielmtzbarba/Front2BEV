import torch
from torch import nn
import torch.nn.functional as F

class PonTrainer(nn.Module):
    def __init__(self, model, criterion, rank):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.rank = rank
    
    def forward(self, images, labels, phase):
        pass