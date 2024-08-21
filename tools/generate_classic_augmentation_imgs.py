import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from src.data.front2bev.utils import process_path


def load_image(image_path, rgb=True):
    """Load an image from the specified path."""
    if rgb:
        image = cv2.imread(image_path)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def flip_image(image):
    """Flip the input image horizontally."""
    flipped_image = cv2.flip(
        image, 1
    )  # 1 for horizontal flip, 0 for vertical flip, -1 for both flips
    return flipped_image


def blur_image(image, kernel_size=(39, 39)):
    """Apply Gaussian blur to the input image."""
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def plot_images(image1, image2):
    """Plot three images in a grid of 1 row and 2 columns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 3 columns

    # Plot first image
    axes[0].imshow(image1)
    axes[0].set_title("original")

    # Plot second image
    axes[1].imshow(image2)
    axes[1].set_title("augmented")

    # Hide axis for all plots
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_rgbd(image, row, method, depth=False):
    """Save the image to the specified path."""
    dirs = row.split("/")
    dirs[len(dirs) - 3] = method
    outdir, name = dirs[:-1], dirs[len(dirs) - 1]
    path = "/".join(outdir)
    try:
        os.makedirs(path)
    except:
        pass
    imgpath = os.path.join(path, name)
    cv2.imwrite(imgpath, image)
    return imgpath


def save_bev(image, row, method):
    """Save the image to the specified path."""
    dirs = row.split("/")
    dirs[len(dirs) - 4] = method
    outdir, name = dirs[:-1], dirs[len(dirs) - 1]
    path = "/".join(outdir)
    try:
        os.makedirs(path)
    except:
        pass
    imgpath = os.path.join(path, name)
    image.save(imgpath)
    return imgpath


def augment(dataset, method):
    df = dataset.copy()
    for i, row in tqdm(dataset.iterrows()):
        rgb_img = load_image(row[0])
        bev_img = Image.open(row[1])
        depth_img = load_image(row[2])
        if method == "blur":
            aug_img = blur_image(rgb_img)
            aug_depth_img = blur_image(depth_img)
            rgbpath = save_rgbd(aug_img, row[0], method)
            bevpath = save_bev(bev_img, row[1], method)
            depthpath = save_rgbd(aug_depth_img, row[2], method)
        if "flip" in method:
            aug_img = flip_image(rgb_img)
            aug_depth_img = flip_image(depth_img)
            rgbpath = save_rgbd(aug_img, row[0], method)
            bev_aug = ImageOps.mirror(bev_img)
            bevpath = save_bev(bev_aug, row[1], method)
            depthpath = save_rgbd(aug_depth_img, row[2], method)
        df.loc[0, i] = rgbpath
        df.loc[1, i] = bevpath
        df.loc[2, i] = depthpath
    return df


csv_dataset_path = "datasets/f2b-mini-rgbd/front2bev-train.csv"
#dataset_path = "/media/aisyslab/dan/datasets/Front2BEV-RGBD/"
# dataset_path = "/home/dan/Data/datasets/Front2BEV-RGBD/"
dataset_path = "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Front2BEV-RGBD/"


def main():
    # Change dataset relative paths to absolute paths
    df = pd.read_csv(csv_dataset_path, header=None)
    df = process_path(df, dataset_path, 5, "traffic")

    aug_dataset = augment(df, "flip")
    df_aug = pd.concat([df, aug_dataset], ignore_index=True)

    aug_dataset = augment(df, "blur")
    dfout = pd.concat([df_aug, aug_dataset], ignore_index=True)

    dfout.to_csv("test.csv", header=False, index=False)
    return


if __name__ == "__main__":
    main()
