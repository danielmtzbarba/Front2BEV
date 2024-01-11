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

def compare_morph(sem_img):
    bev_gt = bev.postprocess(sem_img, bev_cls[N_CLASSES],
                          N_CLASSES, morph=False)
    bev_morph = bev.postprocess(sem_img, bev_cls[N_CLASSES],
                          N_CLASSES, morph=True)
    
    bev_rgb = bev.bevAsRGB(bev_gt, N_CLASSES, bev_class2color)
    bev_morph_rgb = bev.bevAsRGB(bev_morph, N_CLASSES, bev_class2color)

    graph.compare_images([sem_img, bev_rgb, bev_morph_rgb],
                          title=['Top semantic view','BEV remap','BEV Morphology'])



# --------------------------------------------------------------
ROOT_PATH = "/media/danielmtz/data/datasets/Dan-2023-Front2BEV/"
MAP = 'Town10HD/'
LAYERS = 'traffic/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev/sem"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')

N_CLASSES = 6
test_img = '/media/danielmtz/data/datasets/Dan-2023-Front2BEV/Town06/layers_all/bev/$k/0.png'.replace("$", str(N_CLASSES))

# --------------------------------------------------------------
    

rand_img_path = random.choice(bev_img_paths)

img_path = y_dir + '/201.jpg'
bev_sem = cv2.imread(rand_img_path, 0)

size = (196, 200)

decoded_masks = bev.postprocess(bev_sem, bev_cls[N_CLASSES], size, 
                        bev.resize_img(mask64, size), N_CLASSES, morph=True).transpose((2, 1, 0))

decodedplt = np.transpose(decoded_masks, (2, 1, 0))

encoded_mask = encode_binary_labels(decoded_masks)
#Image.fromarray(encoded_mask.astype(np.int32), mode='I').save('test.png')
#print('Image saved')


# Load image
encoded = to_tensor(Image.open(test_img)).long()  
decoded = decode_binary_labels(encoded, N_CLASSES + 1).numpy().transpose((2, 1, 0))
print(encoded.shape, decoded.shape)

labels, mask = decoded[:, :, 0:-1], decoded[:, :, N_CLASSES]
encoded = encoded.numpy().transpose((2, 1, 0))
plot_encoded_masks(encoded)
plot_class_masks(labels, mask)

# Convert to a torch tensor
#

#plt.imshow(img)
#plot_class_masks(decoded_masks)
#plot_class_masks(decoded)

plt.show()




