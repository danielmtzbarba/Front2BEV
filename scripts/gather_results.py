# ----------------------------------------------------------------------------- import os import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dan.utils import load_pkl_file
# -----------------------------------------------------------------------------

cols=['experiment','config' , 'model', 'opt', 'weight_mode', 'miou', 'macc', 'highest_epoch', 'train_time']


def get_log_data(exp, logdir):
    args = exp.split('-')
    model = 'ved' if 'ved' in args else 'pon' 
    config = 'aug' if 'aug' in exp else 'traffic'
    weight = args[len(args)-1] 
    opt = 'adam' if 'adam' in exp else 'sgd'
    print("\n==> Experiment:", exp )
    log_file = load_pkl_file(f'{logdir}/{exp}/{exp}.pkl')
    try:
        miou = log_file['epochs']['val_miou'] 
        macc = log_file['epochs']['val_macc']
    except:
        miou = log_file['epochs']['val_iou'] 

    macc = [np.mean(acc) for acc in log_file['epochs']['val_acc']] 
    
    idx = np.argmax(miou)
    data = [exp, config, model, opt, weight, miou[idx], macc[np.argmax(macc)], idx, np.mean(log_file['epochs']['train_time'])]
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
def main():
  df =  get_experiments(logdir)
  df.drop(['experiment'], axis=1, inplace=True)

  print(df[df['opt']=='adam'].sort_values(by=['miou'], ascending=False))

if __name__ == '__main__':
    main()
