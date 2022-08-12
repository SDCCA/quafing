import os
import warnings
import numpy as np
from quafing.multipdf.multipdf_collection import MultiPdfCollection


class Embedder(object):

	def __init__(self,mdpdf_collection=None):

		if not mdpdf_collection is None:
			if isinstance(mdpdf_collection, MultiPdfCollection):
				self._mdpdfc = mdpdf_collection
			else:
				self._mdpdfc = None
				warnings.warn(f'Warning: specified multi dimensional pdf collection oes not have expected type.')

	    self._dmatrix = None
	    

	def user_supplied_matrix(self,A):

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

    def set_embedding_parameters(self):
    	raise NotImplementedError(
            "Class %s doesn't implement set_embedding_parmeters()"% self.__class__.__name__ )

    def embed(self):
        raise NotImplementedError(
            "Class %s doesn't implement embed()"% self.__class__.__name__ )


