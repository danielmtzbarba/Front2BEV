import deeplab
from deeplab.train import train
from deeplab.utils.configs import get_configuration

if __name__ == "__main__":
    config = get_configuration(train=True)
    train(config)
