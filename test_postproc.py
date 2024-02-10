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
encoded_img = 'test.png' 
encoded_img = '/media/dan/dan/datasets/Dan-2024-Front2BEV/Town01/scene_1/flip/bev/5k/27.png'
N_CLASSES = 5
size = (200, 196)

#_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')
#rand_img_path = random.choice(bev_img_paths)

# --------------------------------------------------------------

pixel_count_total = {key:value for (key,value) in enumerate([0 for i in range(N_CLASSES)])}
pixel_count_total["N"] = 0

fov_mask = bev.resize_img(mask64, size)
bev_img, decoded_masks, pixel_count = bev.postprocess(bev_sem, bev_cls[N_CLASSES], size, fov_mask, 
                         N_CLASSES, morph=True, display=False)

for (key,value) in pixel_count.items():
    pixel_count_total[key] += value

print(pixel_count_total)
# --------------------------------------------------------------
# Save encoded labels
encoded_masks = encode_binary_labels(decoded_masks.transpose((2, 1, 0))).transpose()
plt.imshow(encoded_masks)
plt.show()
Image.fromarray(encoded_masks.astype(np.int32), mode='I').save('test.png')

# Load and decode labels
encoded = to_tensor(Image.open(encoded_img)).long() 
decoded = decode_binary_labels(encoded, N_CLASSES + 1)
print(encoded.shape, decoded.shape)

# --------------------------------------------------------------
# Plot labels
labels = decoded[:-1]
print(labels.shape, fov_mask.shape)

encodedplt = encoded.numpy().transpose(1, 2, 0)
labelsplt = labels.numpy().transpose(1, 2, 0)

plt.imshow(encodedplt)
plt.show()
#plot_encoded_masks(encodedplt)
plot_class_masks(labelsplt, fov_mask)
plt.show()
masked = bev.masks2bev(labels.unsqueeze(0), torch.from_numpy(fov_mask).unsqueeze(0)) 
#plt.imshow(masked.numpy().transpose(1, 2, 0))
#plt.show()

# --------------------------------------------------------------
