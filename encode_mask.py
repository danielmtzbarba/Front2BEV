import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
from src.utils import configs

from src.data.dataloader import get_dataloaders

import matplotlib.pyplot as plt
import numpy as np
from src.utils.visualize import colorise

import torchvision.transforms.functional as F

from src.factory.builder import Builder
import dan.utils.graph as graph
# -----------------------------------------------------------------------------

def compare_bev_pred(mask):

    plt.imshow(mask)
    plt.show()


def binarize_mask(tensor):
    vmin, vmax = 0, 1
    tensor = tensor.detach().cpu().float()
    return (tensor - vmin) / (vmax - vmin)

def decode_class_mask(labels):
    for i, ch in enumerate(labels):
        class_masks = binarize_mask(ch)
        labels[i, :, :] = class_masks
    return class_masks, np.argmax(labels, 0)

def mask_img(labels, mask, num_class):
    labels[mask] = num_class
    return labels

def tensor2image(img):
    return np.array(F.to_pil_image(img))

def plot_class_masks(class_masks):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 10))

    for i in range(len(class_masks)):
        msk = np.reshape(class_masks[i, :, :].numpy().astype('int'), (196, 200))
        ax.ravel()[i].imshow(msk)

def main(rank: int, config: object):

    dataloaders = get_dataloaders(config)
    
    data = dataloaders['train']
    for batch in data:
        image, calib, labels, mask = batch
        
        #image = tensor2image(image[0])

        class_masks, bev = decode_class_mask(labels)

        plot_class_masks(class_masks)

        #bev = mask_img(class_mask, np.reshape(np.invert(mask[0].numpy()), (196, 200)), 14)

        builder = Builder(config, 0)
        model_trainer = builder.get_test_objs()
        logits, loss = model_trainer(batch, "val")

        scores = logits.cpu().sigmoid() > config.score_thresh
        pred_masks, pred_bev = decode_class_mask(scores)
        #pred = mask_img(preds, np.reshape(np.invert(mask[0].numpy()), (196, 200)), 14)
        plot_class_masks(pred_masks)

        plt.show()
        break
    

if __name__ == '__main__':

    config = configs.get_configuration()
    logdir = configs.create_experiment(config, None)
    
    main(0, config)


