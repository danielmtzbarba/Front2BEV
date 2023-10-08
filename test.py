import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")


import argparse

from tests.Front2BEV.dev import args
from utils.dataloader import get_dataloaders

from dan.utils.torch import set_deterministic

from vae_test import test_model


def set_console_args(console_args):
    config = console_args.mapconfig
    n = console_args.kclasses
    test_name = f"F2B_VAE_{config}_{n}k"
    args.test_name = test_name
    args.n_classes = int(n)

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

def main():
    """
    Main method
    """

    argparser = argparse.ArgumentParser(description='Front2BEV Test')
    
    argparser.add_argument('-c', '--mapconfig', help='Map Config')

    argparser.add_argument('-k', '--kclasses', help='K classes')
    
    console_args = argparser.parse_args()
    args = set_console_args(console_args)
    set_deterministic(args.seed)

    print("\n", args.test_name)

    print("\n", args.test_csv_path)

    test_model(args)

if __name__ == '__main__':
    main()