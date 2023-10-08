import pandas as pd

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dan.utils.torch.transforms import Transforms
from dan.utils.torch.datasets import Front2BEVDataset
from utils.transforms import Rescale, ToTensor

def get_f2b_dataloader(root_path: str, csv_path: str, batch_size: int, n_workers = 8, distributed=False):

    # Change dataset relative paths to absolute paths
    df = pd.read_csv(csv_path, header=None)
    df = df.apply(lambda path: (root_path + path))

    dataset = Front2BEVDataset(df, transform=Transforms([Rescale((256, 512)), ToTensor()]))
    if distributed:
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = False, shuffle = False,
                                 num_workers=0, sampler = DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers=n_workers)

    return dataloader
