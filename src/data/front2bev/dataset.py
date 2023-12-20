# ------------------------------------------------------------------------------------------------------
from skimage.io import imread
import numpy as np

import torch
from torchvision import transforms

from torch.utils.data import Dataset

from src.utils.transforms import Rescale, Normalize
# ------------------------------------------------------------------------------------------------------

class Transforms(transforms.Compose):
    def __init__(self, transforms):
        self.transforms = transforms

calib = np.array([[129.4754,   0.0000, 132.6751],
                 [  0.0000, 460.3568, 274.0784],
                 [  0.0000,   0.0000,   1.0000]])

class Front2BEVDataset(Dataset):

    def __init__(self, df, image_size, num_class):

        self.samples = df
        self.image_size = image_size
        self.num_class = num_class
        self.transform = Transforms([Rescale(self.image_size),
                                      Normalize()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_sem_mask(idx)
        sample = {'image': image.float(),
                   'label': label.float(),
                   'calib': torch.from_numpy(calib).float()
                  }

        return sample
    
    def load_image(self, idx):
        # Load image
        img = imread(self.samples.iloc[idx, 0])
        img = self.transform(img)
        # Convert to a torch tensor
        return img
    
    def load_sem_mask(self, idx):
        # Load image
        img = imread(self.samples.iloc[idx, 1])
        # Convert to a torch tensor
        return torch.from_numpy(np.array(img))
# ------------------------------------------------------------------------------------------------------