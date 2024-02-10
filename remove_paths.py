import pandas as pd

def replace_paths(csv_file, old_substring, new_substring):
    """Lee un archivo CSV con rutas de archivos y reemplaza una parte de esas rutas."""
    # Lee el archivo CSV en un DataFrame de Pandas
    df = pd.read_csv(csv_file, header=None)

    # Aplica el reemplazo a la primera columna del DataFrame
    df[0] = df[0].str.replace(old_substring, new_substring)
    df[1] = df[1].str.replace(old_substring, new_substring)
    # Devuelve la lista de rutas actualizadas
    return df 

# Ejemplo de uso:
csv_file = 'datasets/Dan-2024-Front2BEV-ClassicAug/front2bev-test.csv'
old_substring = 'Town01/scene_11/'
new_substring = 'Town02/scene_1/'
updated_paths = replace_paths(csv_file, old_substring, new_substring)

updated_paths.to_csv(csv_file, header=False, index=False)
print(updated_paths.head())

