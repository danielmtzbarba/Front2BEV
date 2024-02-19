
import os, time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import process_time
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from src.utils.transforms import Rescale, Normalize
from src.data.utils import decode_binary_labels

import src.data.front2bev.bev as bev
from src.data.front2bev.bev_classes import bev_cls, bev_class2color
from src.data.front2bev.utils import mask64

from src.utils.visualize import plot_class_masks, plot_encoded_masks
# ------------------------------------------------------------------------------------------------------

class Transforms(transforms.Compose):
    def __init__(self, transforms):
        self.transforms = transforms

def plot(imgs):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imgs[0])
    ax[1].imshow(imgs[1])
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

def plot_results(logits, batch, thresh):
    # plot
    image, calib, labels, mask = batch

    scores = logits.cpu().sigmoid() > thresh


    decoded_labels = decode_class_mask(labels)
    labels = mask_img(decoded_labels, np.reshape(np.invert(mask.numpy()), (196, 200)), 14)

    decoded_preds = decode_class_mask(scores)
    #preds = mask_img(decoded_preds, np.reshape(np.invert(mask.numpy()), (196, 200)), 14)

    plot([labels, decoded_preds])

def show_results(batch, mask):
#    bev_map = bev.masks2bev(batch, mask).numpy().transpose(1, 2, 0)
    print(batch.shape, mask.shape)
    plot_class_masks(batch.squeeze(0).numpy().transpose(1, 2, 0), mask.transpose())
    plt.show()

def load_image(im_path):
    return Image.open(im_path)

def preprocess(img_path, transforms):
    img = load_image(img_path)
    img = transforms(np.array(img)).float()
    return img.unsqueeze(0)

def transform(model, im_path, calib, config):

    inference_time = 0
    calib = torch.from_numpy(calib).float().unsqueeze(0).to(0)
    fov_mask = bev.resize_img(mask64, (196, 200))
    transforms = Transforms([Rescale(config.img_size),
                                      Normalize()])
    # Set model to evaluate mode
    model.eval()  
    with torch.set_grad_enabled(False):
        start_time = time.time() 
        # forward
        image = preprocess(im_path, transforms).to(0)

        logits = model(image, calib)
        scores = logits.cpu().sigmoid() > 0.7 
        print(logits)
        inference_time += time.time() - start_time 
       # plot_results(logits, batch, 0.5)

    print("Mean inference time: ", inference_time)        
    show_results(scores, fov_mask) 
    return logits        

