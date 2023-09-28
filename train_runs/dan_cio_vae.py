import pandas as pd
from torch.utils.data import DataLoader

from dan.utils.torch.datasets import Front2BEVDataset
from dan.utils.torch.transforms import Transforms

from models.VAE.data_loader import ToTensor, Rescale

restore_ckpt = False
ckpt_path = '__ckpts/Dan-2023-Front2BEV/front2bev_3k.pth.tar'

n_epochs = 1
batch_size = 4
n_workers = 2
n_classes = 4

# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
ROOT_PATH = "D:/Datasets/"

train_csv_path = "__datasets/Dan-2023-Front2bev/front2bev-train.csv"
val_csv_path = "__datasets/Dan-2023-Front2bev/front2bev-val.csv"

df_train = pd.read_csv(train_csv_path, header=None)
df_train = df_train.apply(lambda path: (ROOT_PATH + path))

df_val = pd.read_csv(val_csv_path, header=None)
df_val = df_val.apply(lambda path: (ROOT_PATH + path))
 
# Define dataloaders
train_set = Front2BEVDataset(df_train, transform=Transforms([Rescale((256, 512)), ToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

val_set = Front2BEVDataset(df_val, transform=Transforms([Rescale((256, 512)), ToTensor()]))
val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=n_workers)

dataloaders = {'train': train_loader, 'val': val_loader}