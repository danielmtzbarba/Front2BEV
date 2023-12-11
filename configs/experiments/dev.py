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
    'n_gpus': 1,

    'num_epochs':  3,
    'batch_size': 4,
    'n_workers': 4,

    'class_weights': None,
    'ignore_class': True,

    # new config
    'model': 'ved',
    'num_class': 0,
    'map_config': "",
    "num_workers": 1,

    'dataset_root': "/media/dan/data/OneDrive/Datasets/Dan-2023-Front2BEV/",
    'csv_path': 'src/datasets/Front2BEV-debug/',
    'logdir': 'logs/',

}
