import numpy as np
from scipy.integrate import quad
from quafing.distance.base_distance import InformationDistancePiecewise

def _choose_hellinger_func(dim):
    if dim == '1d':
        info_dist = hellinger_1d
    elif dim == 'nd':
        """
        TODO implement higher dimensional hellinger distance
        """
        info_dist = hellinger_nd
    else:
        raise RuntimeError(
            f'invalid distance specification {dim}')
    return info_dist



def hellinger_1d(pdf1,pdf2,is_discrete=False, bbox=(-np.inf,np.inf)):
        if is_discrete:
            return discrete_hellinger_1d(pdf1,pdf2)
        else:
            return continuous_hellinger_1d(pdf1,pdf2,bbox=bbox)


def discrete_hellinger_1d(p1,p2):
    """ Computes the Hellinger distance between p1 and p2.

    Args:
        p1 : dict with keys as values and values as probabilities
        p2 : dict with keys as values and values as probabilities

    Returns: The computed Hellinger distance.

    """
    if not isinstance(p1, dict):
        raise RuntimeError(
            "p1 is not a dict")
    if not isinstance(p2, dict):
        raise RuntimeError(
            "p1 is not a dict")	
    
    #assert np.sum(p1.values()) == 1.0, "Sum %f not equal to one!" % np.sum(p1.values())
    #assert np.sum(p2.values()) == 1.0, "Sum %f not equal to one!" % np.sum(p2.values())

    k1 = list(p1.keys())
    k2 = list(p2.keys())

    tot_sum = 0.0
    for k in k1:
        if k in k2:
            tot_sum += (np.sqrt(p1[k]) - np.sqrt(p2[k])) ** 2
        else:
            tot_sum += p1[k]
    non_k1 = [p for p in k2 if p not in k1]
    for k in non_k1:
        tot_sum += p2[k]

    return 2 * np.sqrt(tot_sum)

def continuous_hellinger_1d(p1,p2,bbox=(-np.inf,np.inf)):
    """ Computes the Hellinger distance of two continuous PDFs.

    Args:
        p1 (function): PDF1 to compute Hellinger distance.
        p2 (function): PDF2 to compute Hellinger distance.
        bbox (2-tuple): Integration bounds.

    Returns: The computed Hellinger distance.

    """
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 2 and bbox[0] < bbox[1]
    assert hasattr(p1, '__call__')
    assert hasattr(p2, '__call__')

    def hell(x, p1, p2):
        return (np.sqrt(p1(x)) - np.sqrt(p2(x))) ** 2

    integral = quad(lambda x: hell(x,p1,p2), bbox[0], bbox[1])
    return np.sqrt(integral[0])

def hellinger_nd(is_discrete=False):
    """
    TO DO
    """
    if is_discrete:
        return discrete_hellinger_nd()
    else:
         return continuous_hellinger_nd()

def discrete_hellinger_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd discrete hellinger distance not yet implemented')

def continuous_hellinger_nd():
    """
    TO DO
    """
    raise NotImplementedError(
        'nd continuous hellinger distance not yet implemented')

    


class HellingerDistance(InformationDistancePiecewise):

    def __init__(self,mdpdfs1, mdpdfs2, pwdist=None, dims=None):
        super().__init__(mdpdfs1,mdpdfs2,pwdist)
        if dims is None:
            dims = self._auto_dims()
        self._set_dist(dims)


    def _set_dist(self, dims):
        if isinstance(dims,str):
            self._info_dist = _choose_hellinger_func(dims)
        elif isinstance(dims,list):
            funcs = [ _choose_hellinger_func(dim) for dim in dims]
            self._info_dist = funcs    


