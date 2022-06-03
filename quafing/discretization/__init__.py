import os

from .bayesian_blocks_discretizer import BayesianBlockDiscretizer as bbd

discretizers = {
	'BayesianBlocks':bbd
	}

def get_discretizer(data, colmetadata, method):
	_check_method(method)
	discretizer = discretizers[method]
	return discretizer(data, colmetadata)

def _check_method(method):
    if method not in discretizers:
		raise NotImplementedError(
			"Discretization method %s unknown. Supported methods are: %s" % (method, ', '.join(discretizers.keys())))
