# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------
import cv2
import numpy as np
from src.utils import configs

from src.factory.builder import Builder

from dan.utils.torch import set_deterministic

from src.utils.inferer import transform 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
ROOT_PATH = "/media/danielmtz/data/datasets/Dan-2024-Front2BEV/"
MAP = 'Town01/'
SCENE = 'scene_7/'
LAYERS = 'traffic/'

x_dir = ROOT_PATH + MAP + SCENE + LAYERS + "rgb/"
y_dir = ROOT_PATH + MAP + SCENE + LAYERS + "bev/"

rgb_img_path = x_dir + '10.jpg'
sem_img_path = y_dir + 'sem/10.jpg'

encoded_img = 'test.png' 
encoded_img = '/media/dan/dan/datasets/Dan-2024-Front2BEV/Town01/scene_1/flip/bev/5k/27.png'

calib = np.array([[129.4754,   0.0000, 132.6751],
             [  0.0000, 460.3568, 274.0784],
             [  0.0000,   0.0000,   1.0000]])
# -----------------------------------------------------------------------------

def main(config):
    builder = Builder(config, 0)
    model = builder.get_test_objs().model
    transform(model, rgb_img_path, calib, config)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    config = configs.get_configuration(False)
    set_deterministic(config.seed)

    try:
        main(config)
    except Exception as e:
       print(e) 

