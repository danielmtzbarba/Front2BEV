import warnings
warnings.filterwarnings("ignore")

from __tests.F2B_3K_VAE.hyparams.aircv import train_args
#from train_runs.Front2BEV.VAE.aircv import dev_args

from models.VAE.vae_train import train_model
from dan.utils.torch import set_deterministic

if __name__ == '__main__':
    set_deterministic(train_args.seed)
    train_model(train_args)
