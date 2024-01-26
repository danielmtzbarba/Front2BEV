import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from PIL import Image

from torchvision.transforms.functional import to_tensor
# --------------------------------------------------------------
import dan.utils.graph as graph
import src.data.front2bev.bev as bev

from src.data.front2bev.bev_classes import bev_cls, bev_class2color
from src.data.front2bev.utils import mask64

from dan.utils.data import get_dataset_from_path
# --------------------------------------------------------------
from src.utils.visualize import plot_class_masks, plot_encoded_masks
from src.data.utils import encode_binary_labels, decode_binary_labels
# --------------------------------------------------------------
ROOT_PATH = "/media/dan/data/datasets/Dan-2024-Front2BEV/"
MAP = 'Town01/'
SCENE = 'scene_7/'
LAYERS = 'traffic/'

x_dir = ROOT_PATH + MAP + SCENE + LAYERS + "rgb/"
y_dir = ROOT_PATH + MAP + SCENE + LAYERS + "bev/"

img_path = y_dir + 'sem/10.jpg'
print(img_path)

bev_sem = cv2.imread(img_path, 0)
encoded_img = y_dir + '5k/10.png' 
N_CLASSES = 5
size = (196, 200)

#_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')
#rand_img_path = random.choice(bev_img_paths)


fov_mask = bev.resize_img(mask64, size)
decoded_masks = bev.postprocess(bev_sem, bev_cls[N_CLASSES], size, fov_mask, 
                         N_CLASSES, morph=True, display=True)
#decodedplt = np.transpose(decoded_masks, (2, 1, 0))


# --------------------------------------------------------------
# Save encoded labels

#encoded_mask = encode_binary_labels(decoded_masks)
#Image.fromarray(encoded_mask.astype(np.int32), mode='I').save('test.png')
#print('Image saved')

# Load image
#encoded = to_tensor(Image.open(encoded_img)).long()  
#decoded = decode_binary_labels(encoded, N_CLASSES + 1).numpy().transpose((2, 1, 0))
#print(encoded.shape, decoded.shape)

labels, mask = decoded[:, :, 0:-1], decoded[:, :, N_CLASSES]
encoded = encoded.numpy().transpose((2, 1, 0))
#plot_encoded_masks(encoded)
#plot_class_masks(labels, mask)
#plt.show()
# --------------------------------------------------------------






