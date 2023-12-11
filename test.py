import warnings
warnings.filterwarnings("ignore")

from utils.dataloader import get_f2b_dataloaders

from dan.utils.torch import set_deterministic

from utils.vae_test import test_model

# -----------------------------------------------------------------------------
import argparse

def set_console_args():
    
    from configs.experiments.dev import args

    argparser = argparse.ArgumentParser(description='Front2BEV Trainer')
    
    argparser.add_argument('-c','--mapconfig', help='Map Config')

    argparser.add_argument('-k','--kclasses', help='K classes')

    console_args = argparser.parse_args()

    config = console_args.mapconfig
    n = console_args.kclasses

    test_name = f"FRONT2BEV-VED-{config}-{n}k"
    args["name"] = test_name
    args["num_class"] = int(n)

    return args
# -----------------------------------------------------------------------------

from dan.utils import dict2obj

def main():
    args = set_console_args()
    set_deterministic(args["seed"])

    args = dict2obj(args)
    args.distributed = False

    dataloaders = get_f2b_dataloaders(args)

    args.test_loader = dataloaders['test']

    print("\n", args.name)
    test_model(args)

if __name__ == '__main__':
    main()