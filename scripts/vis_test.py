import imageio
import cv2
from time import sleep

import src.data.front2bev.bev as bev 
from src.data.front2bev.bev_classes import bev_cls

from dan.utils.data import get_dataset_from_path

def show_gif(img_paths_list, n_classes, transform=False):
    gif_img_list = []
#    for im_path in img_paths_list:
    for i in range(100):
        im_path = x_dir + f"/{i}.jpg"
        im_path = x_dir + f"/{i}-gt.jpg"
        frame = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1024,1024), interpolation=cv2.INTER_AREA)
        gif_img_list.append(frame)
        cv2.imshow('video',frame)
#        sleep(0.05)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    cv2.destroyAllWindows()
    return gif_img_list
# *****************************************************
ROOT_PATH = "/media/danielmtz/data/datasets/Dan-2024-Front2BEV/"

MAP = 'Town01/'
LAYERS = 'traffic/'
SCENE = 'scene_1/'
N_CLASSES = 5

x_dir = ROOT_PATH + MAP + SCENE + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + SCENE + LAYERS + "bev/5k"

rgb_img_paths, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.png')


x_dir = 'vis-test/gt'
gif = show_gif(rgb_img_paths, N_CLASSES, False)
imageio.mimsave('vis-test/demo-gt-8FPS.gif', gif, fps=8)
