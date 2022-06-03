import os

from .discrete_1d_estimator import DiscreteDensityEstimator1d as dde1d

estimators = {
	'Discrete1D':dde1d
	}

def get_density_estimator(data, colmetadata, method):
	_check_method(method)
	estimator = estimators[method]
	return estimator(data, colmetadata)

def _check_method(method):
    if method not in estimators:
		raise NotImplementedError(
			"Density estimator %s unknown. Supported methods are: %s" % (method, ', '.join(estimators.keys())))