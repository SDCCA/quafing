""" abstract information distance and attending fucntions"""
import os
import types

import warnings
import numpy as np

from quafing.density import discrete_estimators
from quafing.density import continuous_estimators


def _avg_dist(distances):
    """
    return aggregate of piecewise distances as an average distance

    :param distances: list of piecewise distances between the constituents of  multi-dimensional pdf
    """
    return np.average(np.array(distances))

def _rms_dist(distances):
    """
    return aggregate of piecewise distances as an rms distance

    :param distances: list of piecewise distances between the constituents of  multi-dimensional pdf
    """
    return np.sqrt(np.sum(distances ** 2))

def _sum_dist(distances):
    """
    return aggregate of piecewise distances as a sum of distances

    :param distances: list of piecewise istances between the constituents of  multi-dimensional pdf
    """
    return np.array(distances).sum()

def _get_pwdist_func(pwdist="rms"):
    """
    select piecewise distance aggregation function. Valid options are listed in pwdistfuncs

    :param pwdist: str specifying piecewise distance aggregtor function
    :return pwdistfunc: selected aggregator function
    """
    pwdistfuncs = {
        "avg": _avg_dist,
        "sum": _sum_dist,
        "rms": _rms_dist
    }
    if not pwdist in pwdistfuncs.keys():
        raise RuntimeError(
            f'no valid piecewise distance function specified. valid options are {[f for f in pwdistfuncs]}')
    else:
        pwdistfunc = pwdistfuncs[pwdist]
        return pwdistfunc

def _info_dist(pdf1,pdf1meta,pdf2,pdf2meta,func,kw):
    """
    Abstract information distance function wrapper. Wrapper supports injection of specific distance functions into piecewise distance
    calculation, allowing combined use of discrete and continuoius 1d and nd implementaations of an algorithm within a single multi-dimensional
    pdf.

    :param pdf1: probability density function (1d OR nd, discrete OR continuous). E.g. a piecewise pdf of a multi-dimensional pdf
    :param pdf1meta: metadata of pdf1, format as specified in quafing.multipdf.multi_dimensional_pdf _generate_pdf_meta_entry()
    :param pdf2: probability density function (1d OR nd, discrete OR continuous) of same type as pdf1
    :param pdf2meta: metadata of pdf2
    :param func: distance function object to use for calculating distance
    :param kw: dictionary of keyword arguments to pass to func

    """
    if not pdf1meta == pdf2meta:
        raise RuntimeError(
            'mismatch of component pdf metadata')

    if  pdf1meta['method'] in discrete_estimators.keys():
        is_discrete = True
    elif pdf1meta['method'] in continuous_estimators.keys():
        is_discrete = False
    else:
        raise RuntimeError(
            f"{pdf_meta['method']} is not a valid density estimator")
            
    if kw is None:
        dist = func(pdf1,pdf2,is_discrete=is_discrete)
    else:
        dist = func(pdf1,pdf2,is_discrete=is_discrete,**kw)

    return dist


class InformationDistance(object):
    """
    Abstract information distance class. object to compute information distance between
    two multi-dimensional pdfs. Should handle all type derived from MultiDimensionalPdf class.
    """

    def __init__(self, mdpdf1, mdpdf2):
        """
        initialize object aand validate inputs

        :param mdpdf1: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        :param mdpdf2: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        """

        self._mdpdf1 = mdpdf1
        self._mdpdf2 = mdpdf2

        
        self._validate_input()

    def _validate_input(self):
        """
        Validate input by comparing mdpdf types (must match), column metadata, and length of constituent metadata
        """

        if type(self._mdpdf1) == type(self._mdpdf2):
            pass
        else:
            raise RuntimeError(
                f"mdpdf type mismatch. mdpdf1: {type(self._mdpdf1)} mdpdf2: {type(self._mdpdf2)} ")

        if self._mdpdf1._colmetadata == self._mdpdf2._colmetadata:
            pass
        else:
            raise RuntimeError(
                'mismatch in mdpdf metadata')

        if len(self._mdpdf1._pdf_meta) == len(self._mdpdf2._pdf_meta):
            pass
        else:
            raise RuntimeError(
                f'mismatch in length of mdpdfs. mdpdf1: {len(self._mdpdf1._pdf_meta)} mdpdf2: {len(self._mdpdf2._pdf_meta)} ')

    def calculate_distance(self):
        """
        calculate information distance between mdpdf1 and mdpdf2
        """
        raise NotImplementedError(
            "Class %s doesn't implement calculate_distance()"% self.__class__.__name__ )


class InformationDistancePiecewise(InformationDistance):
    """
    Abstract information distance class for factorized (all dimensions can be considerd iid variables, all constituent pdfs are 1d)
    and partially factorized (some/all dimensions can/must be considered jointly distributed) multi-dimensional pdfs. Distances are defined
    as aggregate of peicewise distances
    """

    def __init__(self, mdpdf1, mdpdf2, pwdist=None):
        """
        initialize, set piecewise aggregator function

        :param mdpdf1: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        :param mdpsf2: multi-dimesnional pdf of type derived from MultiDimensionalPdf class
        :param pwdist: str specifiying piecewise distance aggregator. One of valid keys in 
                   quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
        """
        super().__init__(mdpdf1,mdpdf2)
        self._pwdistfunc = None
        self._info_dist = None

        self._set_pwdist_func(pwdist)

    def _set_pwdist_func(self, pwdist):
        """
        set pwdistfunc to selected piecewise distance aggregator function

        :param pwdist: str specifiying piecewise distance aggregator. One of valid keys in 
                   quafing.distance.base_distance._get_pwdist_func() pwdistfuncs
        """
        self._pwdistfunc = _get_pwdist_func(pwdist)

    def _auto_dims(self):
        """
        Automaticaly determine dimensionality of the piecewise constituent pdf of a multi-dimensional pdf.
        Uses the data_dimension key of the pdf metadata (for mdpdf1).

        :return dims: list of str or str specifying dimensionality ('1d' or 'nd') of constituent pdfs.
                    a single str is dimensionality is te same for all constituent pdfs  

        """
        pdf_meta = self._mdpdf1.get_pdf_metadata()
        dims = []
        for i,md in enumerate(pdf_meta):
            if md['data_dimension'] == 1:
                dim = '1d'
            elif md['data_dimension'] >= 2:
                dim = 'nd'
            else:
                raise RuntimeError(
                    f"unexpected value for data_dimension {md['data_dimension']}")
            dims.append(dim)

        if all([d == dims[0] for d in dims]):
            dims = dims[0]

        return dims
        
        
    def calculate_distance(self, infinite_indicees = False, kwargs_list = None ):
        """
        Calculate distance between two mdpdfs (mdpdf1 and mdpdf2) as an aggregate of the piecewise information
        distanes of their constituent pdfs. Distance function used is injected by derived classes
        Makes use of InformationDistancePiecewise attributes
        
        :param infinite_indicees: bool (default False). If true return list of indicees indicating piecewise distances with
                                infinite values
        :param kwargs_list: optional; list of dictionaries containing keywords to be used for distance calcultion on piecewise pdfs.
                       if passed must match length of dims, or length of auto_dim output

        :return distance: information distance between mdpdf1 aand mdpdf2, calculated as aggregated piecewise distances
        :return infindicees: optional; list of indicees indicating piecewise distances with
                                infinite values  
        """
        if self._pwdistfunc is None:
            raise RuntimeError(
                "no piecewise distance aggregator specified")
        if self._info_dist is None:
            raise RuntimeError(
                "no distance function specified")

        pdfs1 = self._mdpdf1.get_pdf()
        pdfs1_meta = self._mdpdf1.get_pdf_metadata()
        pdfs2 = self._mdpdf2.get_pdf()
        pdfs2_meta = self._mdpdf2.get_pdf_metadata()

        if isinstance(self._info_dist,list):
            if len(self._info_dist) != len(pdfs1):
                raise RuntimeError(
                    'number of specified distance function does not match number of component pdfs')
            func_list = self._info_dist

            if not all([isinstance(f,types.FunctionType) for f in func_list]):
                raise RuntimeError(
                    'distance function list contains non function types')

            if kwargs_list is None:
                kwargs_list = [None]*len(pdfs1)
            else:
                if isinstance(kwargs_list,list) and len(kwargs_list) == len(self._info_dist):
                    if all([isinstance(kwd,dict) for _,kwd in enumerate(kwargs_list)]):
                        pass 
                    else:
                        raise RuntimeError(
                            'all entries in kwargs_list must be of type dict')
                else:
                    raise RuntimeError(
                        'kwargs_list must be of type list and equal in length to the pdfs provided')

        else:
            if not isinstance(self._info_dist,types.FunctionType):
                raise RuntimeError(
                    'distance function is not function type')
            func_list = [self._info_dist]*len(pdfs1)    
            if kwargs_list is None:
                kwargs_list = [None]*len(pdfs1)
            elif isinstance(kwargs_list,dict): 
                kwargs_list = [kwargs_list]*len(pdfs1)
            else:
                raise RuntimeError(
                    'kwargs_list must be a dict')

        pwdistances = []

        for pdf1,pdf1meta,pdf2,pdf2meta,func,kw in zip(pdfs1, pdfs1_meta, pdfs2, pdfs2_meta, func_list, kwargs_list):
            dist = _info_dist(pdf1,pdf1meta,pdf2,pdf2meta,func,kw)
            pwdistances.append(dist)

        pwdistances = np.array(pwdistances)
        infindicees = [i for i,d in enumerate(pwdistances) if np.isinf(d)]

        distance = self._pwdistfunc(pwdistances)

        if infinite_indicees:
            return distance, infindicees
        else:
            return distance    

            




