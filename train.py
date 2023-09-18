import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from dan.tools.dataset2CSV import dataset2CSV
from dan.torch_utils import get_torch_device

from VAE.vae_train import train_model

import warnings
warnings.filterwarnings("ignore")

device = get_torch_device()

n_epochs = 5
batch_size = 1
n_workers = 1
n_classes = 4

ROOT_PATH = "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2023-CarlaBEV/TOWN01/"
x_dir = ROOT_PATH + "rgb"
y_dir = ROOT_PATH + "map"
csv_output_path = "dataset/Front2BEV/bev-vae-test.csv"

restore_ckpt = False
ckpt_path = 'VAE/__checkpoints/vae_front2bev_checkpoint.pth.tar'

def main(create_csv=False):
    # Use train set for choosing hyper-parameters, and use train+val for final traning and testing
    # train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
    train_csv_path = 'dataset/Front2BEV/bev-vae-test.csv'
    val_csv_path = 'dataset/Front2BEV/bev-vae-test.csv'
    if create_csv:
        train_csv_path = dataset2CSV(csv_output_path, x_dir, y_dir, ".jpg", ".png")
        val_csv_path = dataset2CSV(csv_output_path, x_dir, y_dir, ".jpg", ".png")

    train_model(device, batch_size, n_workers, n_epochs, n_classes, train_csv_path, val_csv_path, ckpt_path, restore_ckpt=False)

if __name__ == '__main__':
    main()
