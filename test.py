import deeplab
from deeplab.test import test
from deeplab.utils.configs import get_configuration

LOGDIR = "/run/media/dan/dan/results/f2b-rgbd"

if __name__ == "__main__":
    config = get_configuration(train=False, logdir=LOGDIR)
    test(config)
