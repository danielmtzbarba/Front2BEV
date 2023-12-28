import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import torch
import math

def colorise(tensor, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    
    tensor = tensor.detach().cpu().float()

    vmin = float(tensor.min()) if vmin is None else vmin
    vmax = float(tensor.max()) if vmax is None else vmax

    tensor = (tensor - vmin) / (vmax - vmin)
    return cmap(tensor.numpy())[..., :3]

def bevAsRGB(bev_img, n_classes, cmap):
    bev_img = bev_img.copy()
    bev_rgb = np.stack((bev_img,)*3, axis=-1)
    for cl in range(n_classes):
        try:
            bev_rgb[bev_img == cl, :] = cmap[cl]
        except:
            continue
    bev_rgb[bev_img == n_classes, :] = (0, 0, 0)
    bev_rgb[31, 32, :] = (25, 126, 0)
    return bev_rgb

def plot_post_pipeline(imgs,  titles=[],figsize=(12, 8)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for i, im in enumerate(imgs):
        ax[i].imshow(im)
        ax[i].set_title(titles[i], fontdict={"fontsize": 20})
    fig.suptitle("BEV ground-truth generation", fontsize=30)

def plot_class_masks(class_masks, fov_mask, titles=[], figsize=(20, 10)):
    print(class_masks.shape, fov_mask.shape)

    w, h, c = class_masks.shape
    ncols = 3
    if c + 1 > ncols:
        nrows = (c + 1) // ncols + ((c + 1)  % ncols > 0)
    else:
        nrows = 1
  
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(c):
        msk = np.reshape(class_masks[:, :, i].astype('int'), (w, h))
        ax.ravel()[i].imshow(msk)
    ax.ravel()[i+1].imshow(fov_mask)
    fig.suptitle("BEV Semantic classes", fontsize=30)

def plot_encoded_masks(img, title='',figsize=(20, 10)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.imshow(img)
    fig.suptitle("Encoded semantic masks", fontsize=30)