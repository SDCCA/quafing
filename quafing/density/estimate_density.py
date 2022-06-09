from quafing.density import get_density_estimator

def get_density_estimate(data, colmetadata, method=method, *args, **kwargs):

    estimator = get_density_estimator(data,colmetadata,method)
    density = estimator.obtain_density(*args,**kwargs)
    return density