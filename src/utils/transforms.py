from skimage import transform
import numpy as np
import torch
from torchvision import transforms

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        return transform.resize(img, self.output_size, mode='constant',
                                 preserve_range=False, anti_aliasing=False)

class Normalize(object):
    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        if img.shape[0] == 3: 
            return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(torch.from_numpy(img))
        if img.shape[0] == 4: 
            return transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
                                         std=[0.229, 0.224, 0.225, 0.25])(torch.from_numpy(img))
