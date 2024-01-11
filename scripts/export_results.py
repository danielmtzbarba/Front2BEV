import warnings
warnings.filterwarnings("ignore")

from dan.utils import load_pkl_file
from dan.utils.torch import set_deterministic

import scripts.plot_train_res as graph
from src.utils import configs


# -----------------------------------------------------------------------------

def main(rank: int, config: object):
    log_file = load_pkl_file(config.logdir + f'/{config.name}/{config.model}/{config.name}.pkl')

    
    ax = graph.plot_train_loss(log_file['batches']['loss'],
                                    ['Variational Encoder Decoder', config.map_config, config.num_class],
                                    save_path=f'{config.logdir}/{config.name}/{config.model}/{config.name}-train.png')
    

    ax = graph.plot_val_metrics(log_file['epochs'],
                                ['Variational Encoder Decoder', config.map_config, config.num_class],
                                save_path=f'{config.logdir}/{config.name}/{config.model}/{config.name}-val.png')




if __name__ == '__main__':

    config = configs.get_configuration()
    logdir = configs.create_experiment(config, None)
    
    set_deterministic(config.seed)

    main(0, config)