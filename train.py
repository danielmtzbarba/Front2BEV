import warnings
warnings.filterwarnings("ignore")

#from train_runs.aircv_train_vae import *
from train_runs.dan_cio_vae import *

from dan.utils.torch import get_torch_device
from VAE.vae_train import train_model

device = get_torch_device()

def main():
    train_model(device, dataloaders, n_epochs, n_classes,
                ckpt_path, restore_ckpt=False)

if __name__ == '__main__':
    main()
