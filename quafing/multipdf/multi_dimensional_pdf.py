""" base multidimensional probability density function class """
import os

import pandas as pd

class MultiDimensionalPDF(object):

	def __init__(self, data, colmetadata):

		self._data = data 
		self._colmetadata = colmetadata

		""" validate input format """
        if not isinstance(data, pd.DataFrame):
        	raise RuntimeError(
        		'raw data input is not of type pandas DataFrame')

        if not len(data.columns) == len(colmetadata):
        	raise RuntimeError(
        		'metadata does not match data (column number mismatch)')

      
