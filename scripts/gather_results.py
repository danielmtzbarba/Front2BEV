# ----------------------------------------------------------------------------- import os import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dan.utils import load_pkl_file
# -----------------------------------------------------------------------------

cols=['experiment','config' , 'model', 'opt', 'weight_mode',
      'ious', 'accs', 'miou', 'macc', 'highest_epoch', 'train_time']

cols_iou = ['non-drivable', 'drivable', 'walkway', 'vehicle', 'pedestrian']

def get_ind_metrics(df, wm, cfg, model):
    df = df[df['model']==model]
    df = df[df['weight_mode']==wm].sort_values(by=['miou'], ascending=False)
    df = df[df['config']==cfg]
    df.drop(['config', 'model', 'weight_mode', 'opt'], axis=1, inplace=True)
    print(df['ious'].to_numpy())
    print(np.round(df['miou'], 4)) 

def get_log_data(exp, logdir):
    args = exp.split('-')
    model = 'ved' if 'ved' in args else 'pon' 
    config = 'aug' if 'aug' in exp else 'traffic'
    weight = args[len(args)-1] 
    opt = 'adam' if 'adam' in exp else 'sgd'
#    print("\n==> Experiment:", exp )
    log_file = load_pkl_file(f'{logdir}/{exp}/{exp}.pkl')
    try:
        ious = log_file['epochs']['val_iou']
        accs = log_file['epochs']['val_acc']
        miou = log_file['epochs']['val_miou'] 
        macc = log_file['epochs']['val_macc']
    except:
        miou = log_file['epochs']['val_iou'] 

    macc = [np.mean(acc) for acc in log_file['epochs']['val_acc']] 
    
    idx = np.argmax(miou)
    data = [exp, config, model, opt, weight, [iou.numpy() for iou in ious[idx]], accs[idx], miou[idx], macc[np.argmax(macc)], idx, np.mean(log_file['epochs']['train_time'])]
    return data

def get_experiments(logdir):
    experiments = os.listdir(logdir)
    data = []
    for exp in experiments:

        try: 
            data.append(get_log_data(exp, logdir))
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(data, columns=cols) 
    return df

logdir = '/media/danielmtz/data/logs/run2' 
logdir = "/home/aircv1/Data/Luis/aisyslab/Daniel/results/run2"

def main():
    df =  get_experiments(logdir)
    df.drop(['experiment'], axis=1, inplace=True)
    df = df[df['opt']=='adam'].sort_values(by=['miou'], ascending=False)
    try:
        df = df['ious'] = df['ious'].numpy()
    except: 
        pass
    df.to_csv('run2.csv')
    print(df)
    get_ind_metrics(df, 'sqrt_inverse', 'aug','pon')


if __name__ == '__main__':
    main()
