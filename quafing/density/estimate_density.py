from quafing.density import get_density_estimator

def get_density_estimate(data, method=None, metadata=None *args, **kwargs):

    estimator = get_density_estimator(data, method, metadata=metadata)
    density = estimator.obtain_density(*args,**kwargs)
    return density