from quafing.discretization import get_discretizer

def discretize(data, colmetadata, method=method,*args,*kwargs):
    """
    perform discretization (binning) of answers/variables with continuos support

    :param data: data to discretize
    :param colmetadata: column wise metadata enabling the selection of continuouus data columns
    :param method: keywword (str) specifyying disretization methodd to use
    :return discretization: discretizaation deterrmined. List of arrrays with bin borders
    """ 
	discretizer = get_discretizer(data,colmetadata,method)
	discretization = discretizer.perform_discretization( *args, *kwargs)

	return discretization