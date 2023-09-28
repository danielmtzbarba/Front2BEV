import pandas as pd

def replace_abs_path(csv_path, old_path, new_path):
    df = pd.read_csv(csv_path, header=None)
    new_df = df.replace(regex=[old_path],value=new_path)
    new_df.to_csv(csv_path, header=False, index=False)
    return new_df