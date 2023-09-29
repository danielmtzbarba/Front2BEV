import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

# Use train set for tuning hyper-parameters.
# Then use train+val for final traning and testing.

TEST_NAME = "F2B_3K_VAE"

args = {
    
    'seed': 8964,

    'restore_ckpt': False,
    'ckpt_path': f'/home/aircv1/Data/Luis/aisyslab/Daniel/Checkpoints/{TEST_NAME}.pth.tar',

    'n_epochs':  10,
    'batch_size': 16,
    'n_workers': 8,

    'n_classes': 3,
    'class_weights': None,
    'ignore_class': True,

    # Dataset absolute path
    'dataset_root_path': "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/",
    # Relative paths to csv datasets
    'train_csv_path':  "__datasets/Dan-2023-Front2bev/front2bev-train.csv",
    'val_csv_path': "__datasets/Dan-2023-Front2bev/front2bev-val.csv",
    'test_csv_path': "__datasets/Dan-2023-Front2bev/front2bev-test.csv",

    'log_path': f"__tests/{TEST_NAME}/train_logs/{TEST_NAME}.pkl"
}

from dan.utils import dict2obj
from __tests.F2B_3K_VAE.hyparams import get_dataloaders
from dan.utils.torch import get_torch_device

args = dict2obj(args)
args.dataloaders = get_dataloaders(args)
args.device = get_torch_device()