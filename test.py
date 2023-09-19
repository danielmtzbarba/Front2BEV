import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from dan.utils import make_folder
from dan.utils.torch import get_torch_device

from VAE.vae_test import test_model

import warnings
warnings.filterwarnings("ignore")
#

device = get_torch_device()

CKPT_PATH = 'VAE/__checkpoints/vae_front2bev_checkpoint.pth.tar'
DATASET_CSV_PATH = 'dataset/Front2BEV/bev-vae-test.csv'
BATCH_SIZE = 2

TEST_DIR = "test_concept"
OUTPUT_PATH = make_folder("__results", TEST_DIR)
make_folder(OUTPUT_PATH / "vis")
#

if __name__ == '__main__':
    test_model(device, CKPT_PATH, DATASET_CSV_PATH, BATCH_SIZE, OUTPUT_PATH)