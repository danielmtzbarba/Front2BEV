from src.data.nuscenes.dataloader import get_ns_dataloaders
from src.data.front2bev.dataloader import get_f2b_dataloaders

def get_dataloaders(config):
    if "nuscenes" in config.train_dataset:
        return get_ns_dataloaders(config)
    else:
        return get_f2b_dataloaders(config)
