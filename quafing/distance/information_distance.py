from quafing.distance import get_ID_measure

def information_distance(mdpdf1,mdpdf2,method=None,pwdist=None,dims=None,kwargs_list=None):
    """
    calculate information distance between two multi-dimensional pdfs

    :param mdpdf1: multi-ddimensional pdf
    :param mddpdf2: multi-dimensional pdf
    :param method: distance measure method to use. Must be one of valid keys in quafing.distance.__init__ ID_messures.
    :param pwdist: str, piecewise distance aggregation method to use. one of valid keys is 
                   quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
    :param dims: optional, str or list of str. Dimensionality of (piecewise) mdpdfs for which (piecewise) distance is being calculated, 1d or nd.
                Defaults to use of auto_dims if dims=None. If str, same value is used for all piecewise distances
    :param kwargs_list: optional; list of dictionaries containing keywords to be used for distance calcultion on piecewise pdfs.
                       if passed must match length of dims, or length of auto_dim output 
    """
    ID = get_ID_measure(mdpdf1,mdpdf2,method=method,pwdist=pwdist,dims=dims)
    infodist = ID.calculate_distance(kwargs_list=kwargs_list)

    return infodist 