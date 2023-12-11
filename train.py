import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
import torch.multiprocessing as mp
import torch.distributed as dist 

from utils.dataloader import get_f2b_dataloaders
from utils.trainer import ddp_setup, Trainer
from utils import get_dataset_weights

from models.VAE import get_vae_train_objs

from dan.utils.torch import set_deterministic
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
    args["map_config"] = config

    weights = get_dataset_weights(console_args)
    args["class_weights"] = weights

    return args
# -----------------------------------------------------------------------------
from dan.utils import dict2obj

def train(rank: int, args):
    args = dict2obj(args)
    
    ddp_setup(rank, args.n_gpus)

    vae = get_vae_train_objs(args.num_class)

    dataloaders = get_f2b_dataloaders(args)

    trainer = Trainer(dataloaders, vae['model'], vae['optimizer'],
                       vae["scheduler"], rank, args)
    trainer.train()
    dist.destroy_process_group()

if __name__ == '__main__':
    args = set_console_args()
    set_deterministic(args["seed"])

    print("\n", args["name"])
    
    mp.spawn(train, args=([args]), nprocs=args["n_gpus"])