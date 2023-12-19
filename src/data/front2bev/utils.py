def process_path(df, root_path, num_class, map_config):
    for _, row in df.iterrows():
        row[0] = root_path + row[0].replace("$config", map_config)
        row[1] = root_path + (row[1].replace("$k", f"{num_class}k")).replace("$config", map_config)
    return df