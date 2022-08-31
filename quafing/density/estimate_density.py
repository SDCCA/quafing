from quafing.density import get_density_estimator

def get_density_estimate(data, method=None, metadata=None, *args, **kwargs):
    """
    get density estimate for data

    :param data: data for which density distribuution is to be estimated. Can be a Pandas Dataframe
                or Series, or a NumP ndarray.
    :param method: Str; method to be used. Supported methods are aas listed on quafing.density.__init__
    :param metadata: keyword, optional; metadata ussed in initializing density estimator
    :param *args: arguments to be passed on to density estimator
    :param **kwargs: keyword arguments to be passed on to density estimator
    
    :return density: density estimate. Valid returs will; vary and may include lists (for discrete data) or functions
                    for continuous data
    """

    estimator = get_density_estimator(data, method, metadata=metadata)
    density = estimator.obtain_density(*args,**kwargs)
    return density