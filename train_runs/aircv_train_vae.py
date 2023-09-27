import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

import os
from pathlib import Path
from torch.utils.data import DataLoader

from dan.tools.dataset2CSV import get_csv_datasets
from dan.utils.torch.datasets import Front2BEVDataset
from dan.utils.torch.transforms import Transforms, ToTensor, Rescale

restore_ckpt = False
ckpt_path = 'VAE/__checkpoints/front2bev_260923.pth.tar'

n_epochs = 10
batch_size = 32
n_workers = 8
n_classes = 6

# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
ROOT_PATH = "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2023-Front2BEV/"
x_dir = ROOT_PATH + "rgb"
y_dir = ROOT_PATH + "map"

csv_output_path = "__dataset/Front2BEV/front2bev.csv"

def get_test_dirs(dataset_path):
    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    tests = []
    for map in maps:
        tests.extend([ Path(f.path) for f in os.scandir(map) if f.is_dir() ])
    
    x_dirs, y_dirs = [], []
    for test in tests:
        x_dirs.append(test / "rgb")
        y_dirs.append(test / "bev2")

    return x_dirs, y_dirs

x_dirs, y_dirs = get_test_dirs(ROOT_PATH)
train_csv_path, val_csv_path, _ = get_csv_datasets(csv_output_path, x_dirs, y_dirs)

# Define dataloaders
train_set = Front2BEVDataset(train_csv_path, transform=Transforms([Rescale((256, 512)), ToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

val_set = Front2BEVDataset(val_csv_path, transform=Transforms([Rescale((256, 512)), ToTensor()]))
val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=n_workers)

dataloaders = {'train': train_loader, 'val': val_loader}