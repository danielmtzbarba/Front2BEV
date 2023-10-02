import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from __tests.Front2BEV.hyparams.vae_3kw import args

from models.VAE.vae_train import train_model
from dan.utils.torch import set_deterministic

if __name__ == '__main__':
    set_deterministic(args.seed)
    train_model(args)
