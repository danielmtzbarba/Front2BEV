# Use train set for tuning hyper-parameters.
# Then use train+val for final traning and testing.

TEST_NAME = "TEST_NAME"
CONFIG = "CONFIG"
N_CLASSES = "N_CLASSES"

args = {
    'name': TEST_NAME,
    'seed': 1596,

    'restore_ckpt': False,
    'save_every': 1,

    'distributed': True,
    'n_gpus': 4,

    'num_epochs':  3,
    'batch_size': 32,
    'n_workers': 16,

    'n_classes': 0,
    'class_weights': None,
    'ignore_class': True,

    # new config
    'model': 'ved',
    'num_class': 3,
    'map_config': "layers_all",
    "num_workers": 8,

    'dataset_root': "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2023-Front2BEV/",
    'csv_path': 'src/datasets/Dan-2023-Front2BEV',
    'logdir': f"/home/aircv1/Data/Luis/aisyslab/Daniel/Logs/",
}