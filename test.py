import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from utils import replace_abs_path

from dan.utils import make_folder
from dan.utils.torch import get_torch_device

from models.VAE.vae_test import test_model

import warnings
warnings.filterwarnings("ignore")
#

device = get_torch_device()

CKPT_PATH = '__ckpts/Dan-2023-Front2bev/front2bev_260923.pth.tar'
BATCH_SIZE = 1
N_CLASSES = 4

TEST_DIR = "front2bev_260923"


if __name__ == '__main__':

    DATASET_CSV_PATH = '__datasets/Dan-2023-Front2bev/front2bev-test.csv'

    abs_path = '/home/aircv1/Data/Luis/aisyslab/Daniel/'
    new_abs = '/media/aisyslab/ADATA HD710M PRO/'

    replace_abs_path(DATASET_CSV_PATH, abs_path, new_abs)

    OUTPUT_PATH = make_folder("__results", TEST_DIR)
    make_folder(OUTPUT_PATH / "vis")

    test_model(device, N_CLASSES, CKPT_PATH, DATASET_CSV_PATH, BATCH_SIZE, OUTPUT_PATH)