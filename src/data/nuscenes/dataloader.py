import os

from src.data.nuscenes.dataset import NuScenesMapDataset, NuScenes
from src.data.nuscenes.splits import *

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def get_ns_dataloader(nuscenes_version, dataroot, label_root, scenes, img_size, num_epochs,
                       batch_size, n_workers = 8, distributed=False):
    
    nuscenes = NuScenes(nuscenes_version,  os.path.expandvars(dataroot))
    dataset = NuScenesMapDataset(nuscenes, label_root, img_size, scenes)

    if distributed:
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = False, shuffle = False,
                                 num_workers=0, sampler = DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers=n_workers)

    return dataloader


def get_ns_dataloaders(config):
    
    
    train_loader = get_ns_dataloader(config.nuscenes_version, config.dataroot, config.label_root, TRAIN_SCENES,
                                     config.img_size, config.num_epochs, config.batch_size, config.num_workers, config.distributed)
    
    val_loader = get_ns_dataloader(config.nuscenes_version, config.dataroot, config.label_root, VAL_SCENES,
                                     config.img_size, config.num_epochs, batch_size=1, n_workers=1, distributed=False)
    
    test_loader = get_ns_dataloader(config.nuscenes_version, config.dataroot, config.label_root, TEST_SCENES,
                                     config.img_size, config.num_epochs, batch_size=1, n_workers=1, distributed=False)
        
    return {"train": train_loader, "val": val_loader, "test": test_loader}
    
  

