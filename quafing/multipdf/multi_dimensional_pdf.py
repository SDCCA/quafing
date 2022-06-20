""" base multidimensional probability density function class """
import os

import pandas as pd

class MultiDimensionalPDF(object):

    def __init__(self, data, colmetadata):

        self._data = data 
        self._colmetadata = colmetadata
        self._pdfs = None

        """ validate input format """
        if not isinstance(data, pd.DataFrame):
            raise RuntimeError(
                'raw data input is not of type pandas DataFrame')

        if not len(data.columns) == len(colmetadata):
            raise RuntimeError(
                'metadata does not match data (column number mismatch)')

    def get_pdfs(self):
    	return self._pdfs 

    def get_metadata(self):
        return self._colmetadata  
