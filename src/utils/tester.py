from tqdm import tqdm
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


from src.utils.eval import metric_eval_bev
import src.utils.bev as bev

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

def test(model, dataloader, config):

    # Set model to evaluate mode
    model.model.eval()  

    acc = 0.0
    iou = 0.0

    # Iterate over data.
    for batch in tqdm(dataloader):
         
        with torch.set_grad_enabled(False):
            # forward
            logits, loss = model(batch, "val")

            # metrics
            metrics = model.metrics(logits, batch)
            acc += metrics['acc']
            iou += metrics['iou']

            plot_results(logits, batch, 0.5)
            break
        
    print("\nTest acc: ", acc / len(dataloader))
    print("\nTest mIoU: ", iou / len(dataloader))
    