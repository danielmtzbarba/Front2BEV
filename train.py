import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

# -----------------------------------------------------------------------------
import torch.multiprocessing as mp
import torch.distributed as dist 

from utils.dataloader import get_f2b_dataloader
from utils.trainer import ddp_setup, Trainer
from utils import get_dataset_weights

from models.VAE import get_vae_train_objs

from dan.utils.torch import set_deterministic
# -----------------------------------------------------------------------------
import argparse

def set_console_args():
    
    from tests.Front2BEV.dev import args

    argparser = argparse.ArgumentParser(description='Front2BEV Trainer')
    
    argparser.add_argument('-c','--mapconfig', help='Map Config')

    argparser.add_argument('-k','--kclasses', help='K classes')

    console_args = argparser.parse_args()

    config = console_args.mapconfig
    n = console_args.kclasses

    test_name = f"F2B_VAE_{config}_{n}k"
    args["test_name"] = test_name
    args["n_classes"] = int(n)

    args["res_path"] = args["res_path"].replace("TEST_NAME", test_name)
    args["ckpt_path"] = args["ckpt_path"].replace("TEST_NAME", test_name)
    args["log_path"] =  args["log_path"].replace("TEST_NAME", test_name)

    args["train_csv_path"] = args["train_csv_path"].replace("CONFIG", config)
    args["val_csv_path"] = args["val_csv_path"].replace("CONFIG", config)
    args["test_csv_path"] = args["test_csv_path"].replace("CONFIG", config)

    args["train_csv_path"] = args["train_csv_path"].replace("N_CLASSES", n)
    args["val_csv_path"] = args["val_csv_path"].replace("N_CLASSES", n)
    args["test_csv_path"] = args["test_csv_path"].replace("N_CLASSES", n)

    weights = get_dataset_weights(console_args)
    args["class_weights"] = weights

    return args
# -----------------------------------------------------------------------------
from dan.utils import dict2obj

def train(rank: int, args):
    args = dict2obj(args)
    
    ddp_setup(rank, args.n_gpus)

    vae = get_vae_train_objs(args.n_classes)

    train_loader = get_f2b_dataloader(args.dataset_root_path, args.train_csv_path,
                                        args.batch_size, n_workers=args.n_workers,
                                        distributed = args.distributed)
    
    val_loader = get_f2b_dataloader(args.dataset_root_path, args.val_csv_path,
                                    batch_size = 1, n_workers = 1, distributed = False)
    
    dataloaders = {"train": train_loader, "val": val_loader}

    trainer = Trainer(dataloaders, vae['model'], vae['optimizer'],
                       vae["scheduler"], rank, args)
    trainer.train()
    dist.destroy_process_group()

if __name__ == '__main__':
    args = set_console_args()
    set_deterministic(args["seed"])

    print("\n", args["test_name"])
    print("\n", args["train_csv_path"])
    
    mp.spawn(train, args=([args]), nprocs=args["n_gpus"])