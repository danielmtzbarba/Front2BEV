import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
from src.utils import configs

import torch.multiprocessing as mp
import torch.distributed as dist 

from src.factory.builder import Builder
from src.utils.dataloader import get_f2b_dataloaders

import src.utils.trainer as train
from dan.utils.torch import set_deterministic
# -----------------------------------------------------------------------------

def main(rank: int, config: object):
    if config.distributed:
        # Setup training gpu thread
        train.ddp_setup(rank, config.num_gpus)

    # Get model, optimzer and lr_scheduler
    builder = Builder(config, rank)
    model, optimizer, lr_scheduler, criterion = builder.get_train_objs()

    dataloaders = get_f2b_dataloaders(config)

    trainer = train.Trainer(
        dataloaders=dataloaders,
        model_trainer=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        gpu_id=rank,
        config=config
    )

    trainer.train()

    if config.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':

    config = configs.get_configuration()
    logdir = configs.create_experiment(config, None)
    
    set_deterministic(config.seed)

    if config.distributed:
        mp.spawn(main, args=([config]), nprocs=config.num_gpus)
    else:
        main(0, config)