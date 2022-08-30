""" Cossine distance class and algorithm implementations """

import numpy as np
from scipy.integrate import quad
from quafing.distance.base_distance import InformationDistancePiecewise

def _choose_cosine_func(dim):
    """
    Choose which implementation of the symmetrised KL divergence distance to employ (1d or nd)

    :param dim: str indicating dimensionality of pdfs for which distaance is to be calculated
    :return info_dist: information distance function using symmetrised KL divergence algorithm, either for 1d or nd 
    """
    if dim == '1d':
        info_dist = cosine_1d
    elif dim == 'nd':
        """
        TODO implement higher dimensional cosine distance
        """
        info_dist = cosine_nd
    else:
        raise RuntimeError(
            f'invalid distance specification {dim}')
    return info_dist



def cosine_1d(pdf1,pdf2,is_discrete=False, bbox=(-np.inf,np.inf)):
    """
    wrapper for 1d implementations of the cosine distance between two 1d pdfs

    :param pdf1: 1d pdf
    :param pdf2: 1d pdf
    :param is_discrete: bool (default False). if True pdfs are discrete valued
    :param bbox: optional, tuple. bounding box (range) for the numerical integration of the continuous valued (is_discrete = False) pdf
    :return : discrete or continuous 1d cosine distance 

    """
    if is_discrete:
        return discrete_cosine_1d(pdf1,pdf2)
    else:
        return continuous_cosine_1d(pdf1,pdf2,bbox=bbox)


def discrete_cosine_1d(p1,p2):
    """ Computes the cosine (great-circle) distance between p1 and p2.

    Args:
        p1 : dict with keys as values and values as probabilities
        p2 : dict with keys as values and values as probabilities

    Returns: The computed cosine distance.

    """
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)

    # Only keys that are present in both p1 and p2 give contribution
    tot_sum = 0.0
    for k in p1.keys():
        if k in p2.keys():
            tot_sum += np.sqrt(p1[k] * p2[k])
    return np.arccos(tot_sum)

def continuous_cosine_1d(p1,p2,bbox=(-np.inf,np.inf)):
    """ Computes the cosine (great-circle) distance of two continuous PDFs.

    Args:
        p1 (function): PDF1 to compute the cosine distance.
        p2 (function): PDF2 to compute the cosine distance.
        bbox (2-tuple): Integration bounds.

    Returns: The computed cosine distance.

    """
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 2 and bbox[0] < bbox[1]
    assert hasattr(p1, '__call__')
    assert hasattr(p2, '__call__')

    def cos(x, p1, p2):
        return np.sqrt(p1(x) * p2(x))

    integral = quad(lambda x: cos(x, p1, p2), bbox[0], bbox[1])
    return np.arccos(integral[0])

def cosine_nd(is_discrete=False):
    """
    TO DO
    """
    if is_discrete:
        return discrete_cosine_nd()
    else:
         return continuous_cosine_nd()

def discrete_cosine_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd discrete cosine distance not yet implemented')

def continuous_cosine_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd continuous cosine distance not yet implemented')

    


class CosineDistance(InformationDistancePiecewise):
    """
    class to compute cosine distance between mdpdfs
    """

    def __init__(self,mdpdfs1, mdpdfs2, pwdist=None, dims=None):
        """
        Initialize cosine distance object

        :param mdpdf1: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        :param mdpdf2: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        :param pwdist: str specifiying piecewise distance aggregator. One of valid keys in 
                   quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
        :param dims: optional; list of str or str specifying dimensionality ('1d' or 'nd') of constituent pdfs.
                    a single str is dimensionality is te same for all constituent pdfs

        """
        super().__init__(mdpdfs1,mdpdfs2,pwdist)
        if dims is None:
            dims = self._auto_dims()
        self._set_dist(dims)


    def _set_dist(self, dims):
        """
        Inject appropriate distance funtion in class methods

        :param dims: list of str or str specifying dimensionality ('1d' or 'nd') of constituent pdfs.
                    a single str is dimensionality is te same for all constituent pdfs
        """
        if isinstance(dims,str):
            self._info_dist = _choose_cosine_func(dims)
        elif isinstance(dims,list):
            funcs = [ _choose_cosine_func(dim) for dim in dims]
            self._info_dist = funcs    


