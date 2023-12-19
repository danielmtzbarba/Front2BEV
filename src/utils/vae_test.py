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

def test_model(model, dataloader, config):

    # Set model to evaluate mode
    model.eval()  

    acc = 0.0
    iou = 0.0

    # Iterate over data.
    for temp_batch in tqdm(dataloader):
        batch_rgb = temp_batch['image'].float().to(0)
        batch_map_gt = temp_batch['label'].long().to(0)

        # forward
        with torch.set_grad_enabled(False):
            pred_map, _, _ = model(batch_rgb, False)
            bev_gt = batch_map_gt.cpu().numpy().squeeze()
            bev_nn = np.reshape(
                        np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)),
                                    axis=3), [64, 64])
            
            temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, config.num_class)
            acc += temp_acc
            iou += temp_iou

      #  plot([bev.mask_img(bev_nn, bev.mask64, 3), bev_gt])
    
    print("\nTest acc: ", acc / len(dataloader))
    print("\nTest mIoU: ", iou / len(dataloader))
    