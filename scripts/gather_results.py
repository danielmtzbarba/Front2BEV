# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dan.utils import load_pkl_file
# -----------------------------------------------------------------------------

cols=['experiment','config' , 'model', 'opt', 'weight_mode', 'miou', 'macc', 'highest_epoch']
df = pd.DataFrame([], columns=cols) 

def get_log_data(exp, logdir):
    data = []
    print("\n==> Experiment:",  logdir)
    log_file = load_pkl_file(f'{logdir}/{exp}.pkl')
    miou = log_file['epochs']['val_iou'] 
    acc =  
    mean_acc = [np.mean(acc) for acc in log_file['epochs']['val_acc']] 

def get_experiments(logdir):
    experiments = os.listdir(logdir)
    for exp in experiments:
        args = exp.split('-')
        model = 'ved' if 'ved' in args else 'pon' 
        aug = True if 'aug' in exp else False 

        if aug:
            data = [exp, 'augmented', args[2], 'sgd', args[4], , model]
    print(experiments)


logdir = '/media/danielmtz/data/results/' 
def main():
    get_experiments(logdir)

if __name__ == '__main__':
    main()
