# Use train set for tuning hyper-parameters.
# Then use train+val for final traning and testing.

TEST_NAME = "F2B_3K_VAE"

train_args = {
    
    'seed': 8964,

    'restore_ckpt': False,
    'ckpt_path': '__ckpts/Dan-2023-Front2BEV/front2bev_3k.pth.tar',

    'n_epochs':  1,
    'batch_size': 1,
    'n_workers': 1,

    'n_classes': 3,
    'class_weights': None,
    'ignore_index': False,

    # Dataset absolute path
    'dataset_root_path': "E:/Datasets/",
    # Relative paths to csv datasets
    'train_csv_path':  "__datasets/Dan-2023-Front2bev/front2bev-train.csv",
    'val_csv_path': "__datasets/Dan-2023-Front2bev/front2bev-val.csv",
    'log_path': f"__tests/{TEST_NAME}/train_logs/{TEST_NAME}.pkl"
}

from dan.utils import dict2obj
from __tests.F2B_3K_VAE.hyparams import get_dataloaders
from dan.utils.torch import get_torch_device

train_args = dict2obj(train_args)
train_args.data_loaders = get_dataloaders(train_args)
train_args.device = get_torch_device()
