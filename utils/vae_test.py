from tqdm import tqdm
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from models.VAE import VAE

from dan.utils.torch import load_model
from utils.eval import metric_eval_bev
import tools.bev as bev

def plot(imgs):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imgs[0])
    ax[1].imshow(imgs[1])
    plt.show()  

def test_model(args):
    model = VAE(k_classes=args.num_class)
    model = load_model(model, os.path.join(args.logdir, f"{args.name}.pth.tar"))
    model = model.to(0)

    # Set model to evaluate mode
    model.eval()  

    acc = 0.0
    iou = 0.0

    # Iterate over data.
    for temp_batch in tqdm(args.test_loader):
        batch_rgb = temp_batch['rgb'].float().to(0)
        batch_map_gt = temp_batch['map'].long().to(0)

        # forward
        with torch.set_grad_enabled(False):
            pred_map, _, _ = model(batch_rgb, False)
            bev_gt = batch_map_gt.cpu().numpy().squeeze()
            bev_nn = np.reshape(
                        np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)),
                                    axis=3), [64, 64])
            
            temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, args.num_class)
            acc += temp_acc
            iou += temp_iou

      #  plot([bev.mask_img(bev_nn, bev.mask64, 3), bev_gt])
    
    print("\nTest acc: ", acc / len(args.test_loader))
    print("\nTest mIoU: ", iou / len(args.test_loader))
    