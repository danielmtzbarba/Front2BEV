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

from src.data.front2bev.bev_classes import bev_cls, bev_cls_real, bev_class2color
from src.data.front2bev.utils import mask64

from dan.utils.data import get_dataset_from_path

# --------------------------------------------------------------
from src.utils.visualize import plot_class_masks, plot_encoded_masks
from src.data.utils import encode_binary_labels, decode_binary_labels
# --------------------------------------------------------------

# print(np.unique(bev_sem))
# plt.imshow(bev_sem)
# plt.show()


# --------------------------------------------------------------
def bev_pixel_count(num_class=5, pixel_count={}):
    pixel_count_total = {
        key: value for (key, value) in enumerate([0 for i in range(num_class)])
    }
    pixel_count_total["N"] = 0

    for key, value in pixel_count.items():
        pixel_count_total[key] += value

        print(pixel_count_total)

    return pixel_count_total


def post_proc_bev(size, bev_sem, class2color, num_class=5):
    fov_mask = bev.resize_img(mask64, size)
    fov_mask = np.ones_like(fov_mask)

    (bev_img, decoded_masks, pixel_count) = bev.postprocess(
        bev_sem, class2color, size, fov_mask, num_class, morph=True, display=True
    )

    return bev_img, decoded_masks, pixel_count


# --------------------------------------------------------------
def save_encoded_masks(decoded_masks):
    # Save encoded labels
    encoded_masks = encode_binary_labels(decoded_masks.transpose((2, 1, 0))).transpose()
    Image.fromarray(encoded_masks.astype(np.int32), mode="I").save("test.png")
    plt.imshow(encoded_masks)
    plt.show()


def loadndecode(encoded_maks_path, num_class):
    # Load and decode labels
    encoded = to_tensor(Image.open(encoded_maks_path)).long()
    decoded = decode_binary_labels(encoded, num_class + 1)
    print(encoded.shape, decoded.shape)
    labels = decoded[:-1]
    return encoded, labels


# --------------------------------------------------------------


def plot_labels(encoded, labels, fov_mask):
    # Plot labels
    encodedplt = encoded.numpy().transpose(1, 2, 0)
    labelsplt = labels.numpy().transpose(1, 2, 0)
    #
    plt.imshow(encodedplt)
    plt.show()
    plot_encoded_masks(encodedplt)
    plot_class_masks(labelsplt, fov_mask)
    plt.show()


def plot_bev_gt(labels, fov_mask):
    masked = bev.masks2bev(labels.unsqueeze(0), torch.from_numpy(fov_mask).unsqueeze(0))
    plt.imshow(masked.numpy().transpose(1, 2, 0))
    plt.show()


def plot_rgbd(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    img2 = cdf[img]
    cv2.imwrite("/home/dan/Data/figures/CyS2024/RGBD.png", img2)


#    plt.imshow(img2, cmap="binary_r")
#    plt.axis("off")
#    plt.show()


# --------------------------------------------------------------

# ROOT_PATH = "/media/dan/data/datasets/Dan-2024-F2B-Autominy/"
ROOT_PATH = "/home/dan/Data/datasets/Front2BEV-RGBD/"
MAP = "Town02/"
SCENE = "scene_1/"
LAYERS = "traffic/"

bev_dir = ROOT_PATH + MAP + SCENE + LAYERS + "bev/"
depth_dir = ROOT_PATH + MAP + SCENE + LAYERS + "rgbd/31.jpg"
img_path = bev_dir + "sem/31.jpg"

num_class, size = 5, (200, 196)

if __name__ == "__main__":
    print(img_path)
    bev_sem = cv2.imread(img_path, 0)
    depth_img = cv2.imread(depth_dir, 0)
    plot_rgbd(depth_img)
#    encoded_img_path = "test.png"
# bev_gt, decoded, pixel_count = post_proc_bev(size, bev_sem, bev_cls[num_class])
