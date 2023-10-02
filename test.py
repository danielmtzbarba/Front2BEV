import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from __tests.Front2BEV.hyparams.vae_3k import args

from dan.utils.torch import set_deterministic

from models.VAE.vae_test import test_model

if __name__ == '__main__':
    set_deterministic(args.seed)
    test_model(args)
