import os

from .hellinger_distance import HellingerDistance as hd
from .cosine_distance import CosineDistance as cd
from .kl_divergence_distance import SymKLDivDistance as skld

ID_measures = {
    'hellinger': hd,
    'cosine': cd,
    'kl': skld
}

def get_ID_measure(mdpdf1, mdpdf2, method=None, pwdist=None, dims=None):
    """
    Return initialized information distance object of specified type for 2 mdpdfs 

    :param mdpdf1: multidimensional pdf
    :param mdpdf2: multidimensional pdf
    :param method: str, distance measurement method to use. One of vaid keys ID_measures
    :param pwdist: str, piecewise distance aggregation method to use. one of valid keys is 
                   quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
    :param dims: optional, str or list of str. dimensionality of (piecewise) mdpdfs for which (piecewise) distance is being calculated, 1d or nd.
                     defaults to use of auto_dims if dims=None. if str same is used for all piecewise distances
    :return ID_measure: initialized information distance object of specified type for 2 mdpdfs 
    """
    if method is None:
        raise RuntimeError(
            "specification of information distance measure required")
    if pwdist is None:
        raise RuntimeError(
            'specifiation of piecewise distance aggregation function required')

    _check_distance_measure_method(method)
    ID_measure = ID_measures[method]
    return ID_measure(mdpdf1,mdpdf2,pwdist=pwdist,dims=dims)

def _check_distance_measure_method(method):
    """
    check whether distance measure is supported

    :param method: str, distance measurement method to use. One of vaid keys ID_measures
    """
    if method not in ID_measures:
        raise NotImplementedError(
            f"Information distance measure {method} is unknown")