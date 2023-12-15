from os.path import join
import numpy as np
import pandas as pd

from dan.utils import make_folder
from utils import replace_abs_path

def create_dataset_dir(root_path, config, n_classes):
    root_path = make_folder(root_path)
    config_path = make_folder(root_path, config)
    dataset_path = make_folder(config_path, f'{n_classes}k')
    return dataset_path

def dataset2CSV(data_path, csv_path, config, n_classes, n_imgs):
    data = []
    for map in MAPS:
        path = join(data_path, map)
        path = join(path, config)
        for i in range(n_imgs):
            data.append([map, config, join(path, f'rgb/{i}.jpg'),
                         join(path, f'bev/{n_classes}k/{i}.png')])
    
    csv_path = create_dataset_dir(csv_path, config, n_classes)

    dataset = pd.DataFrame(data, columns=['map', 'map_config', 'rgb_path', 'bev_path'])
    csv_file_path = join(csv_path, f'{file_name}_{n_classes}k.csv')
    dataset.to_csv(csv_file_path, header=True, index=False)
    return csv_file_path

def split_dataset(csv_path, train=0.7):
    dataset = pd.read_csv(csv_path)
    train_samples = dataset.groupby("map", group_keys=False).apply(lambda x:x.sample(frac=train))

    test_samples = dataset.iloc[dataset.index.difference(train_samples.index)].reset_index(drop=True)
    val_samples = test_samples.groupby("map", group_keys=False).apply(lambda x:x.sample(frac=0.5))
    test_samples = test_samples.iloc[test_samples.index.difference(val_samples.index)]

    train_samples = train_samples.sample(frac=1, axis=0).reset_index(drop=True)
    val_samples = val_samples.sample(frac=1, axis=0).reset_index(drop=True)
    test_samples = test_samples.sample(frac=1, axis=0).reset_index(drop=True)

    train_samples.to_csv(str(csv_path).replace('.csv', '-train.csv'), columns=['rgb_path', 'bev_path'], header=False, index=False)
    val_samples.to_csv(str(csv_path).replace('.csv', '-val.csv'), columns=['rgb_path', 'bev_path'], header=False, index=False)
    test_samples.to_csv(str(csv_path).replace('.csv', '-test.csv'), columns=['rgb_path', 'bev_path'], header=False, index=False)


file_name = 'front2bev'
csv_root_path = "__datasets/Dan-2023-Front2bev/"
dataset_root_path =  "D:/Datasets/Dan-2023-Front2BEV/"

MAPS = ['Town01', 'Town02', 'Town03', 'Town04',
         'Town05', 'Town06', 'Town07', 'Town10HD']
MAP_CONFIGS = ['layers_all', 'layers_none', 'traffic']
N_CLASSES = [2, 3, 4, 5, 6]

#old_path = r"D:\\Datasets\\"
old_path = r"D:/Datasets/"

for config in ['layers_all', 'layers_none']:
    for n in [2, 3]:
        csv_file_path = dataset2CSV(dataset_root_path, csv_root_path, config, n, n_imgs=2000)
        replace_abs_path(csv_file_path, old_path, "")
        split_dataset(csv_file_path) 
    
config = 'traffic'
for n in [2, 3, 4, 5, 6]:
    csv_file_path = dataset2CSV(dataset_root_path, csv_root_path, config, n, n_imgs=3000)
    replace_abs_path(csv_file_path, old_path, "")
    split_dataset(csv_file_path) 
