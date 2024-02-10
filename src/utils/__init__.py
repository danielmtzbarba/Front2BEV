from pathlib import Path
import pandas as pd
import ast,os

def get_test_dirs(dataset_path):

    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    map_configs, scenes = [], []

    for map in maps:
        scenes.extend([Path(f.path) for f in os.scandir(os.path.join(dataset_path, map))])
        for scene in scenes:
            map_configs.extend([ Path(f.path) for f in os.scandir(scene) if f.is_dir() ])
    print(map_configs)
    return map_configs 

def replace_abs_path(csv_path, old_path, new_path):
    df = pd.read_csv(csv_path, header=None)
    new_df = df.replace(regex=[old_path],value=new_path)
    new_df = new_df.replace(regex=[r"\\"], value="/")

    new_df.to_csv(csv_path, header=False, index=False)
    return new_df

def replace_paths(csv_file, old_substring, new_substring):
    """Lee un archivo CSV con rutas de archivos y reemplaza una parte de esas rutas."""
    # Lee el archivo CSV en un DataFrame de Pandas
    df = pd.read_csv(csv_file, header=None)

    # Aplica el reemplazo a la primera columna del DataFrame
    df[0] = df[0].str.replace(old_substring, new_substring)
    df[1] = df[1].str.replace(old_substring, new_substring)
    # Devuelve la lista de rutas actualizadas
    return df 
