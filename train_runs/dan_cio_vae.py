import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from dan.tools.dataset2CSV import get_csv_datasets

# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
ROOT_PATH = "C:/Users/Danie/OneDrive/dan/RESEARCH/DATASETS/Dan-2023-CarlaBEV/TOWN01/"
x_dir = ROOT_PATH + "rgb"
y_dir = ROOT_PATH + "map"

csv_dataset_path = "__dataset/Front2BEV/bev-vae.csv"

train_csv_path, val_csv_path, _ = get_csv_datasets(csv_dataset_path, x_dir, y_dir)

restore_ckpt = False
ckpt_path = 'VAE/__checkpoints/vae_front2bev_checkpoint.pth.tar'

n_epochs = 1
batch_size = 2
n_workers = 1
n_classes = 4




