import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
from src.utils import configs

from src.factory.builder import Builder
from src.data.dataloader import get_dataloaders

from dan.utils.torch import set_deterministic
# -----------------------------------------------------------------------------

from src.utils.vae_test import test_model

# -----------------------------------------------------------------------------
def main(config):
    dataloaders = get_dataloaders(config)
    builder = Builder(config, 0)
    model = builder.get_test_objs(config)
    test_model(model, dataloaders['test'], config)
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    config = configs.get_configuration()
    logdir = configs.create_experiment(config, None)
    
    set_deterministic(config.seed)

    print("\n", config.name)

    main(config)