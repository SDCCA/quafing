from quafing.discretization import get_discretizer

def discretize(data, colmetadata, method=None, *args, **kwargs):
    """
    perform discretization (binning) of answers/variables with continuos support

    :param data: data to discretize
    :param colmetadata: column wise metadata enabling the selection of continuous data columns
    :param method: keyword (str) specifyying disretization method to use
    :return discretization: discretization determined. Array of Dicts with bin borders
    """ 
    if not method == None:
        discretizer = get_discretizer(data,colmetadata,method)
        discretization = discretizer.perform_discretization( *args, **kwargs)
    else:
        raise RuntimeError(
            'discretize requires a method to be specified, but none was specified')

    return discretization