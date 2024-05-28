import warnings
warnings.filterwarnings("ignore")

from dan.utils import load_pkl_file
from dan.utils.torch import set_deterministic

import scripts.plot_train_res as graph
from src.utils import configs


# -----------------------------------------------------------------------------

def main(rank: int, config: object):
    logdir = config.logdir.replace("logs/", "logs/run4/") 
    print("\n==> Experiment:",  logdir)
    log_file = load_pkl_file(f'{logdir}/{config.name}.pkl')

    
    ax = graph.plot_train_loss(log_file['batches']['loss'],
                                    [config.model, config.map_config, config.num_class],
                                    save_path=f'{logdir}/train.png')
    

    ax = graph.plot_val_metrics(log_file['epochs'],
                                [config.model ,config.map_config, config.num_class],
                                save_path=f'{logdir}/val.png')



if __name__ == '__main__':

    config = configs.get_configuration(train=False)
    
    set_deterministic(config.seed)

    main(0, config)

