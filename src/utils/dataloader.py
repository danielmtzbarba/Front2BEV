from skimage.io import imread
import pandas as pd
import numpy as np
import os

import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.transforms import Rescale, Normalize

# ------------------------------------------------------------------------------------------------------
class Transforms(transforms.Compose):
    def __init__(self, transforms):
        self.transforms = transforms

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

def process_path(df, root_path, num_class, map_config):
    for _, row in df.iterrows():
        row[0] = root_path + row[0].replace("$config", map_config)
        row[1] = root_path + (row[1].replace("$k", f"{num_class}k")).replace("$config", map_config)
    return df

def get_f2b_dataloader(root_path, csv_path, num_class, map_config,
                       batch_size, n_workers = 8, distributed=False):

    # Change dataset relative paths to absolute paths
    df = pd.read_csv(csv_path, header=None)
    df = process_path(df, root_path, num_class, map_config)
    dataset = Front2BEVDataset(df, (256, 512), num_class)
    if distributed:
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = False, shuffle = False,
                                 num_workers=0, sampler = DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers=n_workers)

    return dataloader

def get_f2b_dataloaders(config):
    csv_path = os.path.join(config.csv_path, 'front2bev.csv')
    
    train_csv_path = csv_path.replace('.csv', '-train.csv')
    val_csv_path = csv_path.replace('.csv', '-val.csv')
    test_csv_path = csv_path.replace('.csv', '-test.csv')


    train_loader = get_f2b_dataloader(config.dataset_root, train_csv_path, config.num_class, config.map_config,
                                        config.batch_size, n_workers=config.num_workers, distributed = config.distributed)
    
    val_loader = get_f2b_dataloader(config.dataset_root, val_csv_path, config.num_class, config.map_config,
                                    batch_size = 1, n_workers = 1, distributed = config.distributed)
    
    test_loader = get_f2b_dataloader(config.dataset_root, test_csv_path, config.num_class, config.map_config,
                                    batch_size = 1, n_workers = 1, distributed = config.distributed)
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}