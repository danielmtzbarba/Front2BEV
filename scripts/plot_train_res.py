import numpy as np
import matplotlib
matplotlib.use('qtagg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def style_fig(ax, title):
    ax.set_title(title, fontsize=35, y =1.01)

    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 7

    ax.tick_params(axis='both', which='major', labelsize=33)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    return ax

def plot_epoch_loss(epoch_dict, title="Title", figsize=(12, 8)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(epoch_dict['epoch'], epoch_dict['mean_train_loss'])
    ax.plot(epoch_dict['epoch'], epoch_dict['mean_val_loss'])
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('mean_loss', fontsize=16)
    plt.show()

def plot_train_loss(batch_loss, title=["Title", "", ""], figsize=(25, 10), save_path=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax = style_fig(ax, title[0])

    plt.suptitle('Train Loss', fontsize=45, fontweight='bold', x=0.5, y = 1.0)

    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.xlabel('Iteration', fontsize=40, fontweight='bold')
    plt.ylabel('Loss (mean)', fontsize=40, fontweight='bold')

    ax.plot(batch_loss, 'k.', markersize=8, label='loss')
    ax.legend(fontsize=28, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    ax.text(0.3, 0.935, f'config:', fontweight='bold', fontsize=30, transform = ax.transAxes)
    ax.text(0.39, 0.935, f'{title[1]}', fontsize=28, transform = ax.transAxes)

    ax.text(0.55, 0.93, f'k-classes:', fontweight='bold', fontsize=30, transform = ax.transAxes)
    ax.text(0.675, 0.93, f'{title[2]}', fontsize=28, transform = ax.transAxes)

    if save_path:
     #   fig.savefig(save_path.replace(".png", ".eps"))
        fig.savefig(save_path)

    return ax

def plot_val_metrics(data, title=["Title", "", ""], figsize=(25, 10), save_path=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax = style_fig(ax, '')

    plt.suptitle('Validation metrics', fontsize=45, fontweight='bold', x=0.5, y = 1.0)

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.xlabel('Epoch', fontsize=40, fontweight='bold')
    plt.ylabel('Score', fontsize=40, fontweight='bold')
#    mean_acc = [np.mean(acc) for acc in data['val_acc']] 
#    ax.plot(mean_acc, '--ko', markersize=8, label='acc')
    ax.plot(data['val_miou'], '--bo', markersize=8, label='iou')


    ax.legend(fontsize=28, loc='upper right', bbox_to_anchor=(0.13, 0.99))
    ax.text(0.3, 1.05, f':', fontweight='bold', fontsize=30, transform = ax.transAxes)
    ax.text(0.3, 1.05, f'config:', fontweight='bold', fontsize=30, transform = ax.transAxes)

    ax.text(0.39, 1.05, f'{title[1]}', fontsize=28, transform = ax.transAxes)
    ax.text(0.39, 1.05, f'{title[1]}', fontsize=28, transform = ax.transAxes)

    ax.text(0.55, 1.05, f'k-classes:', fontweight='bold', fontsize=30, transform = ax.transAxes)
    ax.text(0.675, 1.05, f'{title[2]}', fontsize=28, transform = ax.transAxes)

    if save_path:
       # fig.savefig(save_path.replace(".png", ".eps"))
        fig.savefig(save_path)
    plt.show()       
    return ax
