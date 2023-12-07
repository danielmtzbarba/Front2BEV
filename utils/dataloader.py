import pandas as pd
import os

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dan.utils.torch.transforms import Transforms
from dan.utils.torch.datasets import Front2BEVDataset
from utils.transforms import Rescale, ToTensor
'''
def get_f2b_dataloader(root_path: str, csv_path: str, batch_size: int, n_workers = 8, distributed=False):

    # Change dataset relative paths to absolute paths
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(lambda path: (root_path + path))

    dataset = Front2BEVDataset(df, transform=Transforms([Rescale((256, 512)), ToTensor()]))
    if distributed:
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = False, shuffle = False,
                                 num_workers=0, sampler = DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers=n_workers)

    return dataloader
'''

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
    dataset = Front2BEVDataset(df, transform=Transforms([Rescale((256, 512)), ToTensor()]))
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