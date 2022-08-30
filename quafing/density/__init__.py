import os

from .discrete_1d_estimator import DiscreteDensityEstimator1d as dde1d

discrete_estimators = {
	'Discrete1D':dde1d
}

continuous_estimators = {
}

estimators = dict(discrete_estimators, **continuous_estimators)

def get_density_estimator(data, method, metadata=None):
	"""
    Return instance of of specific density estimator specified by method, already initialized 
    with data and optionally column metadata.

    :param data: data for which density distribution is to be estimated 
    :param method: denity estimation method to be applied 
    :param metadata: optional; column metadata describing columns of data
    :return: instance of the specified density estimator
    """
    _check_density_method(method)
    estimator = estimators[method]
    return estimator(data, metadata=metadata)

def _check_density_method(method):
	"""
    check whether requested density estimation method is supported. Raises error if it isn't supported. 

    :param method: density estimation method being requested
    """
    if method not in estimators:
        raise NotImplementedError(
            "Density estimator %s unknown. Supported methods are: %s" % (method, ', '.join(estimators.keys())))