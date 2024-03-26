import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import process_time
from PIL import Image  

import src.data.front2bev.bev as bev
from src.data.front2bev.bev_classes import bev_class2color 
from src.utils.visualize import bevAsRGB

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

def plot_results(i, logits, batch, thresh):
    # plot
    image, calib, labels, mask = batch

    scores = logits.cpu().sigmoid() > thresh

    decoded_labels = decode_class_mask(labels)
    labels = mask_img(decoded_labels, np.invert(mask.bool().numpy()), 5)

    decoded_preds = decode_class_mask(scores)
    preds = mask_img(decoded_preds, np.invert(mask.bool().numpy()), 5)

    cmap = bev_class2color
    labels =  bevAsRGB(labels.numpy(), 5, cmap)
    preds = bevAsRGB(preds.numpy(), 5, cmap)
    Image.fromarray(labels.astype(np.uint8)).save(f"{i}-gt.jpg")
    Image.fromarray(preds.astype(np.uint8)).save(f"{i}-pred.jpg")
    plot([labels, preds])

def log_metrics(config, confusion):
    print('-' * 50, f'\nResults {config.name}:')
    acc = confusion.accuracy.numpy()
            
    for name, iou_score, ac in zip(config.class_names, confusion.iou, acc):
        print('{:20s} {:.4f} {:.4f}'.format(name, iou_score, ac)) 
    
    print("\nTest IoU: ", confusion.mean_iou)
    print("Test acc: ", np.mean(acc))

def test(tester, dataloader, config):
    inference_time = 0
    # Set model to evaluate mode
    tester.model.eval()  
    i = 0
    for batch in tqdm(dataloader):
         
        with torch.set_grad_enabled(False):
            start_time = process_time()
            # forward
            logits, loss = tester(batch, "test")
            inference_time += process_time() - start_time 
            # metrics
            metrics = tester.metrics(logits, batch)

            plot_results(i, logits, batch, 0.5)
        
    log_metrics(config,  tester.cm)
    print("Mean inference time: ", inference_time/len(dataloader))        
    
