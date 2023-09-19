import warnings
warnings.filterwarnings("ignore")

from dan.utils.torch import get_torch_device
from VAE.vae_train import train_model

from train_runs.aircv_train_vae import *
#from train_runs.dan_cio_vae import *

device = get_torch_device()

def main():
    train_model(device, batch_size, n_workers, n_epochs, n_classes,
                 train_csv_path, val_csv_path, ckpt_path, restore_ckpt=False)

if __name__ == '__main__':
    main()
