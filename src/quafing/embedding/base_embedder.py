""" Abstract embedder"""

import os
import warnings
import numpy as np
from quafing.multipdf.multipdf_collection import MultiPdfCollection


class Embedder(object):
    """
    Abstract embedder class

    discretizers should handle a multi-dimensional pdf collection or a distance matrix
    """

    def __init__(self,mdpdf_collection=None):
        """
        Initialize and check valid type of optional mdpdf_collection.

        :param mdpdf_collection: optional; multi-dimensioonal pdf collection for which to calculate embeddding
        """

        self._mdpdfc = None
        if not mdpdf_collection is None:
           if isinstance(mdpdf_collection, MultiPdfCollection):
                self.load_mdpdf_collection(mdpdf_collection)
        else:
            warnings.warn(f'Warning: specified multi dimensional pdf collection does not have expected type.')

        self._dmatrix = None
        self._embedding_method = None
	    

    def load_mdpdf_collection(self, collection):
        """
        load multi-dimensional pdf collectiion into embedder instance
        Updates self._mdpdfc

        :param collection: multi-pdf collection of type MultiPdfCollection  
        """
        if self._mdpdfc is None:
            if isinstance(collection, MultiPdfCollection):
	            self._mdpdfc = collection 
            else:
                self._mdpdfc = None
                warnings.warn(f'Warning: specified multi dimensional pdf collection does not have expected type.')
        else:
            raise RuntimeError(
                'Embedder already initialized with a multi dimensional pdf collection')           

    def user_supplied_matrix(self,A):
        """
        load a user supplied matrix (a distance or similarity matrix) into embeder instance (instead of a mdpdf collection).
        Embedding can then be calculated for this matrix.
        Updates self._dmatrix

        :param A: finite positive definite square matrix in the form of an np.ndarray
        """

        if self._mdpdfc is not None:
            raise RuntimeError(
                'Embedder already initialized with a multi dimensional pdf collection')
        else:
            if isinstance(A,np.ndarray):
                if len(A.shape) == 2 and A.shape[0] == A.shape[1]:
                    if np.all(np.isfinite(A)) and np.all(A >= 0):
                        self._dmatrix = A
                    else:
                        warnings.warn('All entries must be finite and postive')
                else:
                    warnings.warn('matrix must be two dimensional and square')
            else:
                warnings.warn('specified matrix is not of required type np.ndarray')

    def set_embedding_parameters(self,**kwargs):
        """
        set parameters for embedding algorithm

        :param kwargs: variable lenght keyword arguments to be passed to embeddng algorithm 
        """
        raise NotImplementedError(
            "Class %s doesn't implement set_embedding_parmeters()"% self.__class__.__name__ )

    def embed(self):
        """
        calculate embedding for distance or similarity matrix. provided by mdpdf-collection or user supplied matrix  
        """
        raise NotImplementedError(
            "Class %s doesn't implement embed()"% self.__class__.__name__ )


