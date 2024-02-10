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
    
    trainer = train.Trainer(
            dataloaders=dataloaders,
            model_trainer=model_trainer,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            gpu_id=rank,
            config=config
            )

    trainer.train()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    config = configs.get_configuration()
    set_deterministic(config.seed)

    try:
        main(config)
    except Exception as e:
       print(e) 

