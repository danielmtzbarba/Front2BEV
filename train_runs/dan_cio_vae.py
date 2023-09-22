from torch.utils.data import DataLoader

from dan.tools.dataset2CSV import get_csv_datasets
from dan.utils.torch.datasets import Front2BEVDataset
from dan.utils.torch.transforms import Transforms, ToTensor, Rescale

restore_ckpt = False
ckpt_path = 'VAE/__checkpoints/vae_front2bev_checkpoint.pth.tar'

n_epochs = 1
batch_size = 2
n_workers = 1
n_classes = 4

# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
ROOT_PATH = "C:/Users/Danie/OneDrive/dan/RESEARCH/DATASETS/Dan-2023-CarlaBEV/TOWN01/"
x_dir = ROOT_PATH + "rgb"
y_dir = ROOT_PATH + "map"

csv_output_path = "__dataset/Front2BEV/bev-vae.csv"

train_csv_path, val_csv_path, _ = get_csv_datasets(csv_output_path, x_dir, y_dir)

# Define dataloaders
train_set = Front2BEVDataset(train_csv_path, transform=Transforms([Rescale((256, 512)), ToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

val_set = Front2BEVDataset(val_csv_path, transform=Transforms([Rescale((256, 512)), ToTensor()]))
val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=n_workers)

dataloaders = {'train': train_loader, 'val': val_loader}


