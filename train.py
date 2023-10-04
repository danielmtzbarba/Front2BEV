import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

import argparse
import ast
import pandas as pd

from models.VAE.vae_train import train_model
from dan.utils.torch import set_deterministic

from __tests.Front2BEV.hyparams import get_dataloaders
from __tests.Front2BEV.hyparams.vae import args

def set_console_args(console_args):
    config = console_args.mapconfig
    n = console_args.kclasses
    test_name = f"F2B_VAE_{config}_{n}k"
    args.test_name = test_name
    args.n_classes = n

    args.res_path = args.res_path.replace("TEST_NAME", test_name)
    args.ckpt_path = args.ckpt_path.replace("TEST_NAME", test_name)
    args.log_path = args.log_path.replace("TEST_NAME", test_name)

    args.train_csv_path = args.train_csv_path.replace("CONFIG", config)
    args.val_csv_path = args.val_csv_path.replace("CONFIG", config)
    args.test_csv_path = args.test_csv_path.replace("CONFIG", config)

    args.train_csv_path = args.train_csv_path.replace("N_CLASSES", n)
    args.val_csv_path = args.val_csv_path.replace("N_CLASSES", n)
    args.test_csv_path = args.test_csv_path.replace("N_CLASSES", n)

    args.dataloaders = get_dataloaders(args)
    return args

def get_datset_weights(console_args):
    df_weights = pd.read_csv(f'__datasets/Dan-2023-Front2bev/{console_args.mapconfig}/{console_args.kclasses}k/weights_{console_args.kclasses}k.csv')
    weights_fov_dict = ast.literal_eval(df_weights['fov_weights'][0])
    return [weights_fov_dict[i] for i in range(int(console_args.kclasses))]

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='Front2BEV Trainer')
    
    argparser.add_argument(
        '-c', '--mapconfig',
        help='Map Config')

    argparser.add_argument(
        '-k', '--kclasses',
        help='K classes')
    
    console_args = argparser.parse_args()
    args = set_console_args(console_args)
    set_deterministic(args.seed)
    weights = get_datset_weights(console_args)
    args.class_weights = weights

    print("\n", args.test_name)

    print("\n", args.train_csv_path)

    print("\nfov_weights:", weights)

    train_model(args)


if __name__ == '__main__':
    main()