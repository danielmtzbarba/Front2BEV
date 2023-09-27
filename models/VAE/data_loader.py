import pandas as pd
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OccMapDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.samples = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        rgb = io.imread(self.samples.iloc[item, 0])
        map = io.imread(self.samples.iloc[item, 1])

        sample = {'rgb': rgb,
                   'map': map,
                  }
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = np.expand_dims(sample['map'], 0)

        rgb = rgb.transpose((2, 0, 1))
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(torch.from_numpy(rgb))
        map = torch.from_numpy(map)
        return {'rgb': rgb,
                'map': map
                }


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        rgb = transform.resize(rgb, self.output_size, mode='constant', preserve_range=False, anti_aliasing=False)

        return {'rgb': rgb,
                'map': map
                }


class Img_distro(object):

    def __init__(self, rot_deg, pix_offset):
        self.rot_deg = rot_deg
        self.pix_offset = pix_offset

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        tran_mat = transform.AffineTransform(translation=(0, self.pix_offset))
        shifted = transform.warp(rgb, tran_mat, preserve_range=True)

        rotated = transform.rotate(shifted, self.rot_deg)

        return {'rgb': rotated,
                'map': map
                }



class Normalize(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(rgb)
        return {'rgb': rgb,
                'map': map
                }
