from src.data.nuscenes.dataloader import get_ns_dataloaders
from src.data.front2bev.dataloader import get_f2b_dataloaders

def get_dataloaders(config):
    if config.train_dataset == "front2bev":
        return get_f2b_dataloaders(config)
    if config.train_dataset == "nuscenes":
        return get_ns_dataloaders(config)