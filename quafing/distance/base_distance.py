import os
import types

import warnings
import numpy as np

from quafing.density import discrete_estimators
from quafing.density import continuous_estimators


def _avg_dist(distances):
    return np.average(np.array(distances))

def _rms_dist(distances):
    return np.sqrt(np.sum(distances ** 2))

def _sum_dist(distances):
    return np.array(distances).sum()

def _get_pwdist_func(pwdist="rms"):
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

    def __init__(self, mdpdf1, mdpdf2):

        self._mdpdf1 = mdpdf1
        self._mdpdf2 = mdpdf2

        
        self._validate_input()

    def _validate_input(self):

        if type(self._mdpdf1) == type(self._mdpdf2):
            pass
        else:
            raise RuntimeError(
                f"mdpdf type mismatch. mdpdf1: {self._mdpdf1._type} mdpdf2: {self._mdpdf2._type} ")

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
        raise NotImplementedError(
            "Class %s doesn't implement calculate_distance()"% self.__class__.__name__ )


class InformationDistancePiecewise(InformationDistance):

    def __init__(self, mdpdf1, mdpdf2, pwdist=None):
        super().__init__(mdpdf1,mdpdf2)
        self._pwdistfunc = None
        self._info_dist = None

        self._set_pwdist_func(pwdist)

    def _set_pwdist_func(self, pwdist):
        self._pwdistfunc = _get_pwdist_func(pwdist)

    def _auto_dims(self):
        pdf_meta = self._mdpdfs1.get_pdf_metadata()
        dims = []
        for i,md in enumerate(pdf_meta):
            if len(md['data_dimension']) == 1:
                dim = '1d'
            elif len(md['data_dimension']) >= 2:
                dim = 'nd'
            else:
                raise RuntimeError(
                    f"unexpected value for data_dimension {md['data_dimension']}")
            dims.append(dim)

        if all([d == dims[0] for d in dims]):
            dims = dims[0]

        return dims
        
        
    def calculate_distance(self, infinite_indicees = False, kwargs_list = None ):
        if self._pwdistfunc is None:
            raise RuntimeError(
                "no piecewise distance aggregator specified")
        if self._info_dist is None:
            raise RuntimeError(
                "no distance function specified")

        pdfs1 = self._mdpdfs1.get_pdf()
        pdfs1_meta = self._mdpdfs1.get_pdf_metadata()
        pdfs2 = self._mdpdfs2.get_pdf()
        pdfs2_meta = self._mdpdfs2.get_pdf_metadata()

        if isinstance(self._info_dist,list):
            if len(self._info_dist) != len(pdfs1):
                raise RuntimeError(
                    'number of specified distance function does not match number of component pdfs')
            func_list = self._info_dist

            if not all([isinstance(f,types.FunctionType) for f in func_list]):
                raise RuntimeError(
                    'distance funtion list contains non function types')

            if kwargs_list is None:
                kwargs_list = [None]*len(pdfs1)
            else:
                if not isinstance(kwargs_list,list) and len(kwargs_list) == len(self._info_dist):
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

            




