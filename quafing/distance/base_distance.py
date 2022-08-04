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



class InformationDistance(object):

    def __init__(self, mdpdf1, mdpdf2):

        self._mdpdf1 = mdpdf1
        self._mdpdf2 = mdpdf2

        
        self._validate_input()

    def _validate_input(self):

    	if self._mdpdf1._type == self._mdpdf2._type:
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
		
		
	def calculate_distance(self, infinite_indicees = False, **kwargs ):
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

        pwdistances = []
        

        for i in len(range(pdfs1)):
            if not pdfs1_meta[i] == pdfs2_meta[i]:
                raise RuntimeError(
                    'mismatch of component pdf metadata')

            if  pdfs1_meta[i]['method'] in discrete_estimators.keys():
            	is_discrete = True
            elif pdfs1_meta[i]['method'] in continuous_estimators.keys():
            	is_discrete = False
            else:
            	raise RuntimeError(
            		"pdf_meta['method'] is not a valid density estimator")

            if isinstance(self._info_dist, types.FunctionType):
                dist = self._info_dist(pdfs1[i],pdfs2[i],is_discrete=is_discrete, **kwargs)
            elif isinstance(self._info_dist, list) and isinstance(self._info_dist[i], types.FunctionType):
            	dist = self._info_dist[i](pdfs1[i],pdfs2[i],is_discrete=is_discrete, **kwargs)
            else:
            	raise RuntimeError(
            		'unexpected specification of distance function(s)')

            pwdistances.append(dist)

        pwdistances = np.array(pwdistances)
        infindicees = [i, for i,d in enumerate(pwdistances) if np.isinf(d)]

        distance = self._pwdistfunc(pwdistances)

        if infinite_indicees:
            return distance, infindicees
        else:
            return distance    

            




