from operator import mul
from functools import reduce

from src.models.ved import VariationalEncoderDecoder 
from src.models.rgved import RGVED
from src.models.rgved_2h import RGVED_2H

from src.models.pyramid import PyramidOccupancyNetwork

from src.models.nn.fpn import FPN50
from src.models.nn.topdown import TopdownNetwork
from src.models.nn.pyramid import TransformerPyramid
from src.models.nn.classifier import LinearClassifier, BayesianClassifier


def build_model(config):
    model_name = config.model

    if model_name == 'pon':
        model = build_pyramid_occupancy_network(config)
    elif model_name == 'ved':
        model = build_variational_encoder_decoder(config)
    elif model_name == 'rgved':
        model = build_rgved(config)
    elif model_name == 'rgved_2h':
        model = build_rgved_2h(config)

    else:
        raise ValueError("Unknown model name '{}'".format(model_name))
    return model

def build_variational_encoder_decoder(config):
    return VariationalEncoderDecoder(config.num_class, 
                                     config.ved.bottleneck_dim,
                                     config.map_extents,
                                     config.map_resolution)

def build_rgved(config):
    return RGVED(config.num_class, 
                 config.ved.bottleneck_dim,
                 config.map_extents,
                 config.map_resolution)

def build_rgved_2h(config):
    return RGVED_2H(config.num_class, 
                 config.ved.bottleneck_dim,
                 config.map_extents,
                 config.map_resolution)


def build_pyramid_occupancy_network(config):

    # Build frontend
    frontend = FPN50()

    # Build transformer pyramid
    tfm_resolution = config.map_resolution * reduce(mul, config.topdown.strides)
    transformer = TransformerPyramid(256, config.tfm_channels, tfm_resolution,
                                     config.map_extents, config.ymin, 
                                     config.ymax, config.focal_length)

    # Build topdown network
    topdown = TopdownNetwork(config.tfm_channels, config.topdown.channels,
                             config.topdown.layers, config.topdown.strides,
                             config.topdown.blocktype)
    
    # Build classifier
    if config.bayesian:
        classifier = BayesianClassifier(topdown.out_channels, config.num_class)
    else:
        classifier = LinearClassifier(topdown.out_channels, config.num_class)
    classifier.initialise(config.prior)
    
    # Assemble Pyramid Occupancy Network
    return PyramidOccupancyNetwork(frontend, transformer, topdown, classifier)
