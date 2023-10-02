# Use train set for tuning hyper-parameters.
# Then use train+val for final traning and testing.

TEST_NAME = "F2B_VAE_3K"

args = {
    'test_name': TEST_NAME,
    'res_path': f'__results/{TEST_NAME}/',
    'seed': 1596,

    'restore_ckpt': False,
    'ckpt_path': f'/home/aircv1/Data/Luis/aisyslab/Daniel/Checkpoints/{TEST_NAME}.pth.tar',
    'log_path': f"/home/aircv1/Data/Luis/aisyslab/Daniel/Logs/{TEST_NAME}.pkl",

    'n_epochs':  3,
    'batch_size': 32,
    'n_workers': 16,

    'n_classes': 3,
    'class_weights': None,
    'ignore_class': True,

    # Dataset absolute path
    'dataset_root_path': "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/",
    # Relative paths to csv datasets
    'train_csv_path':  "__datasets/Dan-2023-Front2bev/front2bev-train.csv",
    'val_csv_path': "__datasets/Dan-2023-Front2bev/front2bev-val.csv",
    'test_csv_path': "__datasets/Dan-2023-Front2bev/front2bev-test.csv",
}

from dan.utils import dict2obj
from __tests.Front2BEV.hyparams import get_dataloaders
from dan.utils.torch import get_torch_device

args = dict2obj(args)
args.dataloaders = get_dataloaders(args)
args.device = get_torch_device()