from dan.utils.torch.transforms import Transforms

from models.VAE.data_loader import Rescale, ToTensor

import pandas as pd

from dan.utils.torch.datasets import Front2BEVDataset
from torch.utils.data import DataLoader

def get_dataloaders(args):
    root_path = args.dataset_root_path
    train_csv_path = args.train_csv_path
    val_csv_path = args.val_csv_path

    # Change dataset relative paths to absolute paths
    df_train = pd.read_csv(train_csv_path, header=None)
    df_train = df_train.apply(lambda path: (root_path + path))

    df_val = pd.read_csv(val_csv_path, header=None)
    df_val = df_val.apply(lambda path: (root_path + path))
    # Dataset and Dataloaders
    train_set = Front2BEVDataset(df_train, transform=Transforms([Rescale((256, 512)), ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    val_set = Front2BEVDataset(df_val, transform=Transforms([Rescale((256, 512)), ToTensor()]))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=args.n_workers)

    return {'train': train_loader, 'val': val_loader}
