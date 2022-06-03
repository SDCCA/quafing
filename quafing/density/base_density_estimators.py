""" Abstract density estimator classes 
    abstract base class Densityestimator
    with derived abstract classes for discrete
    and continuous densitty estimates 

"""
import os
import numpy as np
import pandas as pd 

class DensityEstimator(object):
        """
        Abstract density estimator class

        density estimators should handle discrete or continuous real-valued data
        of one or multiple dimensions
        """

    def __init__(self, data, colmetadata):
    	"""
    	perform checks on data type and determine dimensionaality of data

        :param data: data to estimate density for in form of ndarray, pandas DataFrame orr pandas Series
        :param colmetadata: column(-wise) meta data, including data type (continuous etc.) of column(s)
        """  		
   		self._data = data 
    	self._colmetadata = colmetadata
    	#self._dim = None

    	self._check_data_type()
    	#_get_dimensionality()

    def _check_data_type(self):
        if not (isinstance(self._data, np.ndarray) or isinstance(self._data,pd.DataFrame) or isinstance(self._data,pd.Series)):
            raise TypeError(
                'data of type %s not supported' )% (type(self._data))
    """
    def obtain_density_estimate(self):
    	obtain density estimate for data object
    	:return : density_estimate
    
        
        raise NotImplementedError(
        	"Class %s doesn't implement obtain_density_estimate()"% self.__class__.__name__)
    """

class DiscreteDensityEstimator(DensityEstimator):
    """
    Abstract density estimator class for discrete or discretized data

    discrete density estimators should handle dicrete or continuous real-valued data
    (in the latter case a discretization should be supplied) of one or multiple dimensions.

    discrete density estimators should return density estimates as key value pairs
    """

    def _check_discretization_info(self, discrete=True, discretization=None):
        if discrete:
            if discretization == None:
                self._set_class_discretization_info(discrete,discretization)
            else:
                raise RuntimeError(
                    'discretization schemes for intrinsically discrete data are not supported')
        else:
            if self._discretization not None:
                self._set_class_discretization_info(discrete,discretization)
            else:
                raise RuntimeError(
                    'Estimation of discrete densities for non-discrete data requires a user supplied discretization')

    def _set_class_discetization_info(self, discrete, discretization):
        self._discrete = discrete
        self._discretization = discretization

    

class ContinuousDensityEstimator(DensityEstimator):
    """
    Abstract class for continuous density estimatates on continuous 
    data sets

    continuous density estimators should handle continuous real-valued data
    of one or multiple dimensions.

    continuous density estimators should return a function
    """



    


