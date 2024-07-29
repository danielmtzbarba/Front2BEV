from deeplab.train import train 

class Config(object):
    distributed = False

if __name__ == '__main__':
    config = Config()
    train(config)
