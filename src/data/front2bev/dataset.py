# ------------------------------------------------------------------------------------------------------
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from torch.utils.data import Dataset

from src.utils.transforms import Rescale, Normalize
from src.data.utils import decode_binary_labels
# ------------------------------------------------------------------------------------------------------

class Transforms(transforms.Compose):
    def __init__(self, transforms):
        self.transforms = transforms

class Front2BEVDataset(Dataset):
    calib = np.array([[129.4754,   0.0000, 132.6751],
                 [  0.0000, 460.3568, 274.0784],
                 [  0.0000,   0.0000,   1.0000]])

    def __init__(self, df, image_size, output_size, num_class):

        self.samples = df
        self.image_size = image_size
        self.output_size = output_size
        self.num_class = num_class
        
        self.transform = Transforms([Rescale(self.image_size),
                                      Normalize()])
        self.resize_gt = Transforms([Rescale((*output_size, num_class))])
        self.resize_mask = Transforms([Rescale(output_size)])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        labels, mask = self.load_class_masks(idx)
        image = image.float()
        calib = torch.from_numpy(self.calib).float()
    
        return image, calib, labels, mask
    
    def load_image(self, idx):
        # Load image
        img = Image.open(self.samples.iloc[idx, 0])
        img = self.transform(np.array(img))
        # Convert to a torch tensor
        return img
    
    def load_class_masks(self, idx):
        # Load image
        encoded = to_tensor(Image.open(self.samples.iloc[idx, 1])).long()  
        # Decode into mask classes
        decoded = decode_binary_labels(encoded, self.num_class + 1)
        labels, mask = decoded[:-1], ~decoded[-1]
        return labels, mask
# ------------------------------------------------------------------------------------------------------
