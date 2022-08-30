import os

from .bayesian_blocks_discretizer import BayesianBlockDiscretizer as bbd

discretizers = {
    'BayesianBlocks':bbd
    }

def get_discretizer(data, colmetadata, method):
    """
    Return instance of of specific discretizer specified by method, already initialized 
    with data and column metadata.

    :param data: data to be discretized
    :param colmetadata: column metadata describing columns of data
    :param method: discretization method to be applied
    :return: instance of the specified discretizer
    """
    _check_discretization_method(method)
    discretizer = discretizers[method]
    return discretizer(data, colmetadata)

def _check_discretization_method(method):
    """
    check whether requested discretization method is supported. Raises error if it isn't supported. 

    :param method: discretization method being requested being requested
    """
    if method not in discretizers:
        raise NotImplementedError(
            "Discretization method %s unknown. Supported methods are: %s" % (method, ', '.join(discretizers.keys())))
