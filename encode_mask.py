import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
from src.utils import configs

from src.data.dataloader import get_dataloaders

import matplotlib.pyplot as plt
import numpy as np
from src.utils.visualize import colorise

import torchvision.transforms.functional as F

# -----------------------------------------------------------------------------

def plot_mask(mask):
    plt.imshow(mask)
    plt.show()


def binarize_mask(tensor):
    vmin, vmax = 0, 1
    tensor = tensor.detach().cpu().float()
    return (tensor - vmin) / (vmax - vmin)

def decode_class_mask(labels):
    labels = labels[0]
    for i, ch in enumerate(labels):
        class_mask = binarize_mask(ch)
        labels[i, :, :] = class_mask
    return np.argmax(labels, 0)

def mask_img(labels, mask, num_class):
    labels[mask] = num_class
    return labels

def tensor2image(img):
    return np.array(F.to_pil_image(img))

def main(rank: int, config: object):

    dataloaders = get_dataloaders(config)
    
    data = dataloaders['train']
    for batch in data:
        image, calib, labels, mask = batch
        
        image = tensor2image(image[0])

        class_mask = decode_class_mask(labels)
        bev = mask_img(class_mask, np.reshape(np.invert(mask.numpy()), (196, 200)), 14)
        plot_mask(bev)
        break
    

if __name__ == '__main__':

    config = configs.get_configuration()
    logdir = configs.create_experiment(config, None)
    
    main(0, config)


