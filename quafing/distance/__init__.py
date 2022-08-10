import os

from .hellinger_distance import HellingerDistance as hd
from .cosine_distance import CosineDistance as cd
from .kl_divergence_distance import SymKLDivDistance as skld

ID_measures = {
    'hellinger': hd,
    'cosine': cd,
    'kl': skld
}

def get_ID_measure(mdpdfs1, mdpdfs2, method=None, pwdist=None, dims='1d'):
    if method is None:
        raise RuntimeError(
            "specification of information distance measure required")
    if pwdist is None:
        raise RuntimeError(
            'specifiation of piecewise distance aggregation function required')

    _check_distance_measure_method(method)
    ID_measure = ID_measures[method]
    return ID_measure(mdpdfs1,mdpdfs2,pwdist=pwdist,dims=dims)

def _check_distance_measure_method(method):
    if method not in ID_measures:
        raise NotImplementedError(
            f"Information distance measure {method} is unknown")