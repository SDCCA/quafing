import os
import warnings

from quafing.multipdf.multipdf import create_multi_pdf


def create_mdpdf_collection(mdpfType, group_data, group_labels,colmetadata, calculate=True, validate_metadata=False, *args, **kwargs):
    """
    TODO
    """
    mdpdfs = []
    for i, data in enumerate(group_data):
    	mdpdf = create_multi_pdf(mdpdfType,data, colmetadata, calculate=calculate, *args, **kwargs)
        mdpdfs.append(mdpdf)
    mdpdf_collection = MultiPdfCollection(mdpdfs,group_labels, colmetadata, mdpdfType, validate_metadata=validate_metadata)
	return mdpdf_collection

class MultiPdfCollection(object): 

    def __init__(self, collection, labels, metadata, mdpdftype, validate_metadata=True):

    	self._collection = collection
        self._labels = labels
        self._metadata = metadata
        self._mdpftype = mdpdftype
        self._distance_matrix = None
        self._shortest_path_matrix = None
        self._dissimilarity_matrix = None

        self._validate()
        if validate_metadata:
        	self._validate_metadata()

    def _validate(self):

    	if len(self._labels) != len(list(set(self._labels))):
            warnings.warn(
            	"Duplicate labels were passed ")

    	if len(self._collection) != len(self._labels):
    		raise RuntimeError(
    			f"number of mdpdfs in collection ({len(self._collection)}) does not match number of labels ({len(self._labels)})")

        if not all([mdpdf._mdpftype == self._mdpftype for mdpdf in self._collection]):
        	raise RuntimeError(
        		'mpdf types do not match expected type') 
        																																																																																																																																																								
    def _validate_metadata(self):

    	if not all([mdpdf._colmetadata == self._metadata for mdpdf in self._collection]):
    	    raise RuntimeError(
    	    	'mismatch in metadata. mpdf column metadata does not match reference.')

    	if not all([mdpdf._colmetadata == self._collection[0]._colmetadata for mdpdf in self._collection]):
    		raise RuntimeError(
    			'mismatch between column metadata of mdpdfs in collection ')


    def _calculate_all_mdpdfs(self,*args,**kwargs):
   	    mdpdf.caculate_pdf(*args,**kwargs) for mdpdf in self._collection

   	def calculate_distance_matrix(self,method=None,pwdist=None,dims=None,kwargs_list=None):
   	    """
   	    TODO
   	    """
        mdpdfs = self._collection
        dist_matrix = np.zeros((len(mdpdfs), len(mdpdfs)))
        for i in range(len(mdpdfs)):
            for j in range(i):
                if i == j: continue
                ifd = information_distance(mdpdfs1,mdpdfs2,method=method,pwdist=pwdist,dims=dims,kwargs_list=None)
                distances[i, j] = ifd
                distances[j, i] = distances[i, j]

    def caculate_dissimilarity_matrix(self):
        """	
        TODO
        """

   	def calculate_shortest_path_matrix(self):
   	    """
   	    TODO
   	    """

   	def get_distance_matrix(self):
   		if self._distance_matrix is None:
   			self.calculate_distance_matrix()
   		return self._distance_matrix

   	def get_dissimilarity_matrix(self):
   		if self._dissimilarity_matrix is None:
   			self.calculate_dissimilarity_matrix()
   		return self._dissimilarity_matrix

   	def get_shortest_path_matrix(self):
   	    if self._shortest_path_matrix is None:
   			self.calculate_shortest_path_matrix()
   		return self._shortest_path_matrix  

        




