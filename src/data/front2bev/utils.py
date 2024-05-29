from pathlib import Path
import torch
import numpy as np
import cv2

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

def encode_binary_labels(masks):
    w, h, c = masks.shape
    bits = np.power(2, np.arange(c, dtype=np.int32))
    return (np.resize(masks, (c, w, h)).astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)

def process_path(df, root_path, num_class, map_config):
    for _, row in df.iterrows():
        row[0] = root_path + row[0].replace("$config", map_config)
        row[2] = root_path + row[2].replace("$config", map_config)
        row[1] = root_path + (row[1].replace("$k", f"{num_class}k")).replace("$config", map_config)
    return df

def setup_mask():
    mask64 = cv2.imread(str(Path("src/assets") / "_mask64.png"), 0)
    mask = np.zeros_like(mask64)
    fov = mask64 > 128
    mask[fov] = 1
    return mask

mask64 = setup_mask()

def get_FOV_pixels(masked_image, mask, n_classes):
    fov_mask = mask.reshape(-1) != 0
    fov_index = masked_image.reshape(-1) != n_classes
    fov_index = fov_index * fov_mask
    fov_pixels = masked_image.reshape(-1)[fov_index]
    fov_pixels = fov_pixels.reshape(1, -1)
    return fov_pixels

def count_pixels(img, count_dict, n_classes, only_fov=False):
    if only_fov:
        img = get_FOV_pixels(img, mask64, n_classes)

    clss, counts = np.unique(img, return_counts=True)
    for cls, count in zip(clss, counts):
        count_dict[cls] += count

    n_pixels = len(img.flatten())
    return count_dict, n_pixels
