from pathlib import Path
import pandas as pd
import os

from dan.utils import make_folder
from dan.utils.data import get_dataset_from_path

from dan.tools.dataset2CSV import get_csv_datasets
from utils import replace_abs_path

MAP_CONFIGS = ['layers_all', 'layers_none', 'traffic', 'all_configs']
N_CLASSES = [i for i in range(2, 7)]

def create_dataset_dir(root_path, config, n_classes):
    root_path = make_folder(root_path)
    config_path = make_folder(root_path, config)
    dataset_path = make_folder(config_path, f'{n_classes}k')
    return dataset_path

def get_data_dirs(dataset_path, n_classes, partial=False):
    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    tests = []

    if partial:
        for map in maps:
            tests.append(map / partial)
    else:
        for map in maps:
            tests.extend([ Path(f.path) for f in os.scandir(map) if f.is_dir() ])
    
    x_dirs, y_dirs = [], []
    for test in tests:
        x_dirs.append(test / "rgb")
        y_dirs.append(test / "bev" / f"{n_classes}k")

    return x_dirs, y_dirs

def create_test_csv(output_csv_path, x_dirs, y_dirs, old_path):
    train_csv_path, val_csv_path, test_csv_path = get_csv_datasets(output_csv_path, x_dirs, y_dirs)

    replace_abs_path(train_csv_path, old_path, "")
    replace_abs_path(val_csv_path, old_path, "")
    replace_abs_path(test_csv_path, old_path, "")

def get_dataset_dirs(dataset_root_path, config, n_classes,
                      csv_root_path, csv_filename):
    
    if config == 'all_configs':
        x_dirs, y_dirs = get_data_dirs(dataset_root_path, n_classes)
    else:
        x_dirs, y_dirs = get_data_dirs(dataset_root_path, n_classes, config)
        
        csv_path = create_dataset_dir(csv_root_path, config, n_classes)
        dataset_arrays = get_dataset_from_path(x_dirs, y_dirs,
                                                sufix_x=".jpg", sufix_target=".png")
        
        df = pd.DataFrame(dataset_arrays)
        print(dataset_arrays)

        #create_test_csv(os.path.join(csv_path, csv_filename), x_dirs, y_dirs, old_path)

    return  x_dirs, y_dirs


csv_root_path = "__datasets/Dan-2023-Front2bev/"
dataset_root_path =  "D:/Datasets/Dan-2023-Front2BEV/"
file_name = 'front2bev.csv'
old_path = r"D:\\Datasets\\"

config = 'layers_none'
n_classes = 2

x_dirs, y_dirs = get_dataset_dirs(dataset_root_path, config, n_classes,
                                   csv_root_path, file_name)
print(y_dirs)




