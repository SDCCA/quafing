import os

from .discrete_1d_estimator import DiscreteDensityEstimator1d as dde1d

estimators = {
    'Discrete1D':dde1d
    }

def get_density_estimator(data, method, metadata=None):
    _check_density_method(method)
    estimator = estimators[method]
    return estimator(data, metadata=metadata)

def _check_density_method(method):
    if method not in estimators:
        raise NotImplementedError(
            "Density estimator %s unknown. Supported methods are: %s" % (method, ', '.join(estimators.keys())))