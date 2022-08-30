""" Kullback-Leibler divergence distance (symmetrisd) class and algorithm implementations  """

import numpy as np
from scipy.integrate import quad
import warnings
from quafing.distance.base_distance import InformationDistancePiecewise

def _choose_sym_kl_div_func(dim):
    """
    Choose which implementation of the symmetrised KL divergence distance to employ (1d or nd)

    :param dim: str indicating dimensionality of pdfs for which distaance is to be calculated
    :return info_dist: information distance function using symmetrised KL divergence algorithm, either for 1d or nd 
    """
    if dim == '1d':
        info_dist = sym_kl_div_1d
    elif dim == 'nd':
        """
        TODO implement higher dimensional KL distance
        """
        info_dist = sym_kl_div_nd
    else:
        raise RuntimeError(
            f'invalid distance specification {dim}')
    return info_dist

    


def sym_kl_div_1d(pdf1,pdf2,is_discrete=False, bbox=(-np.inf,np.inf),base=None):
    """
    symmetrisation wrapper for 1d KL divergence distance. set SymKL = sqrt(2*min(Dkl(pdf1||pdf2),Dkl(pdf2||pdf1)))
    This has the same convergence properties as Dkl.

    :param pdf1: 1d pdf
    :param pdf2: 1d pdf
    :param is_discrete: bool (default=False), are pdf1 and pdf2 discrete (either intrinsically or by discretization)
    :param bbox: optional; integration domain for calcualtion of continuous KL divergence
    :param base: base of logartihm for KL divergence
    :return: SymKL as defined above
    """
        if is_discrete:
            kl12 = discrete_kl_div_1d(pdf1,pdf2,base=base)
            kl21 = discrete_kl_div_1d(pdf2,pdf1,base=base)
            if np.isinf(kl12) and np.isinf(kl21):
                return np.inf
            else:
                m = 2.0 * np.nanmin([kl12, kl21])
                return np.sqrt(m)
        else:
            kl12 = continuous_kl_div_1d(pdf1,pdf2,bbox=bbox,base=base)
            kl21 = continuous_kl_div_1d(pdf2,pdf1,bbox=bbox,base=base)
            if np.isinf(kl12) and np.isinf(kl21):
                return np.inf
            else:
                m = 2.0 * np.nanmin([kl12, kl21])
                return np.sqrt(m)

            


def discrete_kl_div_1d(p1,p2,base=None):
    """ Returns the KL divergence between two _discrete_ pdfs, defined as a
    dictionary with possible values as keys and probabilities as values.


        KL(p1||p2) = sum_i p1_i * log (p1_i/p2_i)
        Parameters
        ----------
        p1: dict
        The pdf to compute the kl divergence from.
        p2: dict
        The pdf to compute the kl divergence to.
        base: int or None
        None (natural), 2 or 10 base for the logarithm

    returns: The KL-divergence computed.

    """
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)
    # TODO: Fix this
    #assert np.sum(p1.values()) == 1.0
    #assert np.sum(p2.values()) == 1.0
    assert base in (None, 2, 10)

    if base is None:
        log = np.log
    elif base == 2:
        log = np.log2
    else:
        log = np.log10

    v1, v2 = list(p1.keys()), list(p2.keys())
    kl = 0.0
    for k in v1:
        if k in v2:
            if p1[k] == 0:
                continue
            kl += p1[k] * (log(p1[k]) - log(p2[k]))
        else:
            warnings.warn("Support of p2 smaller than p1 in discrete_kl", RuntimeWarning)
            # TODO
            return np.inf
            #kl += 0
    return kl
    

def continuous_kl_div_1d(p1,p2,bbox=(-np.inf,np.inf),base=None):
    """ Computes the continuous KL divergence from p1 to p2, limited to the
    bounding box bbox.

    Args:
        p1 (function): Distribution from
        p2 (function): Distribution to
        bbox (2-tuple): Tuple with integration boundaries.
        base (None, 2 or 10): The base of the logarithm (natural, 2 or 10).

    Returns: The computed KL divergence.

    """
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 2 and bbox[0] < bbox[1]
    assert hasattr(p1, '__call__')
    assert hasattr(p2, '__call__')
    assert base in (None, 2, 10)

    if base is None:
        log = np.log
    elif base == 2:
        log = np.log2
    else:
        log = np.log10

    def kl(x, p1, p2):
        if p1(x) == 0.0:
            return 0.0
        if p2(x) == 0.0:
            warnings.warn("Continuous KL has support for p2 where p1 is zero.", RuntimeWarning)
            # TODO
            return np.inf
            #return 0.0
        return p1(x) * (log(p1(x)) - log(p2(x)))

    return quad(lambda x: kl(x, p1, p2), bbox[0], bbox[1])[0]	
    

def sym_kl_div_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'sym_kl_div_nd() not implemented')

def discrete_kl_div_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd discrete kl divergence distance not yet implemented')

def continuous_kl_div_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd continuous kl divergence distance not yet implemented')

    


class SymKLDivDistance(InformationDistancePiecewise):
    """
    class to compute symmetrised Kullback-Leibler divergence (SymKL) distance between mdpdfs
    """
    def __init__(self,mdpdf1, mdpdf2, pwdist=None, dims=None):
        """
        Initialize SymKL distance object

        :param mdpdf1: multi-dimensional pdf of type derived from MultiDimensionalPdf class
        :param mdpdf2: multi-dimensional pdf of type derived from MultiDimensionalPdf class
        :param pwdist: str specifiying piecewise distance aggregator. One of valid keys in 
                   quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
        :param dims: optional; list of str or str specifying dimensionality ('1d' or 'nd') of constituent pdfs.
                    a single str is dimensionality is te same for all constituent pdfs

        """
        super().__init__(mdpdf1,mdpdf2,pwdist)
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
            self._info_dist = _choose_sym_kl_div_func(dims)
        elif isinstance(dims,list):
            funcs = [ _choose_sym_kl_div_func(dim) for dim in dims]
            self._info_dist = funcs    


