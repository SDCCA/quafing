""" base multidimensional probability density function class """
import os

import pandas as pd

class MultiDimensionalPDF(object):

    def __init__(self):
        self._data = None
        self._colmetadata = None
        self._pdf = None
        self._pdf_meta = None
        self._type = None

    def _import_data(self, data, colmetadata):
        self._data = data 
        self._colmetadata = colmetadata

    def _basic_validation(self):
        """ validate input format """
        if not isinstance(self._data, pd.DataFrame):
            raise RuntimeError(
                'raw data input is not of type pandas DataFrame')

        if not len(self._data.columns) == len(self._colmetadata):
            raise RuntimeError(
                'metadata does not match data (column number mismatch)')

    def calculate_pdf(self):
    	raise NotImplementedError(
    		"Class %s doesn't implement calculatae_pdf()"% self.__class__.__name__ )

    def get_pdf(self):
    	return self._pdf 

    def get_pdf_metadata(self):
    	return self._pdf_meta

    def get_colmetadata(self):
        return self._colmetadata  
