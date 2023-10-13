from pathlib import Path
import pandas as pd
import ast,os

def get_test_dirs(dataset_path):
    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    tests = []
    for map in maps:
        tests.extend([ Path(f.path) for f in os.scandir(map) if f.is_dir() ])
    return tests

def replace_abs_path(csv_path, old_path, new_path):
    df = pd.read_csv(csv_path, header=None)
    new_df = df.replace(regex=[old_path],value=new_path)
    new_df = new_df.replace(regex=[r"\\"], value="/")

    new_df.to_csv(csv_path, header=False, index=False)
    return new_df

def get_dataset_weights(console_args):
    df_weights = pd.read_csv(f'_datasets/Dan-2023-Front2bev/{console_args.mapconfig}/{console_args.kclasses}k/weights_{console_args.kclasses}k.csv')
    weights_fov_dict = ast.literal_eval(df_weights['fov_weights'][0])
    return [weights_fov_dict[i] for i in range(int(console_args.kclasses))]