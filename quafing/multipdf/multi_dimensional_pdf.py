""" base multidimensional probability density function class """
import os

import pandas as pd

class MultiDimensionalPDF(object):
    """
    Abstract class for multidimensional probability density functions (mdpdfs).
    mdpdfs should handle discrete and continuous density estimates (for discrete and continuous data),
    and either support fully factorized, partially factorized, and/or fully joint distrubted data.

    """
    def __init__(self):
        """
        initialize mdpdf object
        """
        self._data = None
        self._colmetadata = None
        self._pdf = None
        self._pdf_meta = None
        self._type = None

    def _import_data(self, data, colmetadata):
        """
        Import data and column metadata into MultiDimensionalPDF object  instance

        :param data: Pandas DataFrame containing multiple columns of data
        :param colmetadata: column metadata. List of dictionaries containing metadata per column.                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        """
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
        """
        calculate multi-dimensional pdf from data
        """
    	raise NotImplementedError(
    		"Class %s doesn't implement calculatae_pdf()"% self.__class__.__name__ )

    def _generate_pdf_meta_entry(self,method, data_labels, data_dimension, discrete, discretization):
        """
        generate metadata entry for constituent pdf of a multi-dimensional pdf. Entry is of form
        {'method':method, 'data_labels':data_labels, 'data_dimension':data_dimension, 'discrete':discrete, 'discretization':discretiztion}

        :param method: method used in estimating pdf. Valid entry of quafing.density.__init__ estimators
        :param data_labels: label(s) of data column(s) used
        :param data_dimension: dimensionality of joint pdf space
        :param discrete: Is data instrinsically discrete
        :param discretization: iscretization employed for continuous data (if applied).
        :return pdf_meta_entry: dictionary with pdf metadata
        """
        return {'method':method, 'data_labels':data_labels, 'data_dimension':data_dimension, 'discrete':discrete, 'discretization':discretiztion}

    def get_pdf(self):
        """
        return derived multi-dimensional pdf
        """
    	return self._pdf 

    def get_pdf_metadata(self):
        """
        return metadata for constituent pdfs of multi-dimensional pdf.
        List of dictionaries. Expected keys are {method: , data_labels: , data_dimension: , discrete: , discretization: }

        :return self._pdf_meta: pdf metadata for constituent pdfs of multi-dimensional pdf
        """
    	return self._pdf_meta



    def get_colmetadata(self):
        """
        return metadata for columns
        """
        return self._colmetadata  
