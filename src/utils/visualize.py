import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import torch

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

def plot_class_masks(class_masks,  titles=[],figsize=(12, 8)):
    w, h, c = class_masks.shape
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=figsize)
    for i in range(c):
        msk = np.reshape(class_masks[:, :, i].astype('int'), (w, h))
        ax.ravel()[i].imshow(msk)

    fig.suptitle("BEV ground-truth generation", fontsize=30)

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

def encode_binary_labels(masks):
    w, h, c = masks.shape
    bits = np.power(2, np.arange(c, dtype=np.int32))
    return (np.resize(masks, (c, h, w)).astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)