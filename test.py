# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
from src.utils import configs

from src.factory.builder import Builder
from src.data.dataloader import get_dataloaders

import src.utils.trainer as train
from dan.utils.torch import set_deterministic

from src.utils.tester import test
# -----------------------------------------------------------------------------

def main(config):
    dataloaders = get_dataloaders(config)
    builder = Builder(config, 0)
    model = builder.get_test_objs()
    test(model, dataloaders['val'], config)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    config = configs.get_configuration(False)
    set_deterministic(config.seed)

    try:
        main(config)
    except Exception as e:
       print(e) 

