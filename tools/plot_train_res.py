from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from dan.utils import load_pkl_file

def style_fig(ax, title):
    ax.set_title(title, fontsize=20, y =0.9)
    plt.suptitle('Variational Auto Encoder', fontsize=28, fontweight='bold', x=0.32, y = 0.94)

    plt.xlabel('Batch', fontsize=20, fontweight='bold')
    plt.ylabel('Loss (mean)', fontsize=20, fontweight='bold')
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 7

    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    ax.yaxis.set_major_locator(MultipleLocator(0.15))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    return ax

def plot_epoch_loss(epoch_dict, title="Title", figsize=(12, 8)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(epoch_dict['epoch'], epoch_dict['mean_train_loss'])
    ax.plot(epoch_dict['epoch'], epoch_dict['mean_val_loss'])
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('mean_loss', fontsize=16)
    plt.tight_layout()

def plot_batch_loss(batch_loss, title="Title", figsize=(8, 4)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.plot(batch_loss, 'k.', markersize=3, label='loss')
    ax.legend()

    ax = style_fig(ax, title)

    return ax
    



logs_path = Path('D:/Logs/Dan-2023-Front2BEV/')

log_file = load_pkl_file(str(logs_path / 'F2B_VAE_3K.pkl'))

ax = plot_batch_loss(log_file['batches']['loss'], 'VAE mean loss')


#ax2 = plot_batch_loss(log_file['batches']['CE_loss'], 'VAE mean loss')

plt.show()