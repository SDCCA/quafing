import numpy as np
from scipy.integrate import quad
from quafing.distance.base_distance import InformationDistancePiecewise

def _choose_cosine_func(dim):
	if dim == '1d':
    	info_dist = cosine_1d
    elif dim == 'nd':
    	"""
    	TODO implement higher dimensional cosine distance
    	"""
    	info_dist = cosine_nd
    else:
        raise RuntimeError(
        	f'invalid distance specification {dims}')
    return info_dist



def cosine_1d(pdf1,pdf2,is_discrete=False, bbox=(-np.inf,np.inf)):
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

def continuous_cosine_1d(p1,p2,bbox=(-np.inf,np.inf))	
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

def cosine_nd():
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

    def __init__(self,mdpdfs1, mdpdfs2, pwdist=None, dims='1d'):
        super().__init__(mdpdfs1,mdpdfs2,pwdist)
        self._set_dist(dims)


    def _set_dist(self, dims):
        if isinstance(dims,str):
    	    self._info_dist = _choose_cosine_func(dims)
        elif isinstance(dims,list)
            funcs = [ _choose_cosine_func(dim) for dim in dims]
            self._info_dist = funcs    


