""" Abstract discretizer """
import os
import numpy as np
import pandas as pd 

class Discretizer(object):
        """
        Abstract discretizer class

        discretizers should handle continuous real-valued data
        """

    def __init__(self, data, colmetadata, method):
    	"""
    	perform checks on data type and determine dimensionaality of data

        :param data: data to discretize in form of ndarray, pandas DataFrame orr pandas Series
        :param colmetadata: column-wise meta data, including data type (continuous etc.) of column
        :param method: string indicating discretization method to be employed.
        """  		
   		self._data = data 
    	self._colmetadata = colmetadata
    	#self._dim = None

    	_check_data_type()
    	#_get_dimensionality()



    def perform_discretization(self):
    	"""
    	create discretization of data object
    	:return : discretization
    	"""
        
        raise NotImplementedError(
        	"Class %s doesn't implement perform_ddiscretization()"% self.__class__.__name__)


    def _check_data_type(self):
    	if not (isinstance(self._data, np.ndarray) or isinstance(self._data,pd.DataFrame) or isinstance(self._data,pd.Series)):
    		raise TypeError(
    			'data of type %s not supported' )% (type(self._data))

    """
    def _get_dimensionality(self):
    	self._data_dim = self._data.ndim
    """
    