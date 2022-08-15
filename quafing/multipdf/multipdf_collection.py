import os
import warnings
import numpy as np
import networkx as nx
import community
import metis

from quafing.distance.information_distance import information_distance



class MultiPdfCollection(object): 

    def __init__(self, collection, labels, metadata, mdpdftype, validate_metadata=True):

        self._collection = collection
        self._labels = labels
        self._metadata = metadata
        self._mdpdftype = mdpdftype
        self.distance_matrix = None
        self._distance_matrix_type = None
        self.shortest_path_matrix = None
        self.dissimilarity_matrix = None

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

        if not all([mdpdf._type == self._mdpdftype for mdpdf in self._collection]):
            raise RuntimeError(
        	    'mdpdf types do not match expected type') 
        																																																																																																																																																								
    def _validate_metadata(self):

        if not all([mdpdf._colmetadata == self._metadata for mdpdf in self._collection]):
            raise RuntimeError(
            'mismatch in metadata. mpdf column metadata does not match reference.')

        if not all([mdpdf._colmetadata == self._collection[0]._colmetadata for mdpdf in self._collection]):
            raise RuntimeError(
            'mismatch between column metadata of mdpdfs in collection ')


    def _calculate_all_mdpdfs(self,*args,**kwargs):
        for mdpdf in self._collection:
            mdpdf.caculate_pdf(*args,**kwargs)

 	
    def calculate_distance_matrix(self,method=None,pwdist=None,dims=None, return_result=False,kwargs_list=None):
        """
        TODO
        """
        mdpdfs = self._collection
        dist_matrix = np.zeros((len(mdpdfs), len(mdpdfs)))
        for i in range(len(mdpdfs)):
            for j in range(i):
                if i == j: continue
                ifd = information_distance(mdpdfs[i],mdpdfs[j],method=method,pwdist=pwdist,dims=dims,kwargs_list=None)
                dist_matrix[i, j] = ifd
                dist_matrix[j, i] = dist_matrix[i, j]

        if return_result:
            return dist_matrix
        else:
            self.distance_matrix = dist_matrix 
            self._distance_matrix_type = method

    def caculate_dissimilarity_matrix(self):
        """	
        TODO
        """

   	
    def calculate_shortest_path_matrix(self, dist_matrix=None, return_result=False):
        """ Takes the distance matrix and computes the shortest path matrix
        between all pairs of units in the groups.

        Args:
            A (numpy square matrix): A distances matrix (all entries positive)

        Returns: A matrix with shortest distances.

        """
        if not dist_matrix is None:
            A = dist_matrix
        else:
            A = self.distance_matrix


        assert isinstance(A, np.ndarray)
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
        assert np.all(A >= 0)

        g = nx.from_numpy_matrix(A)
        paths = dict(nx.all_pairs_dijkstra_path_length(g))
        new_mat = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(i):
                new_mat[i, j] = new_mat[j, i] = paths[i][j]

        if return_result:
            return new_mat
        else:
            self.shortest_path_matrix = new_mat


    def get_distance_matrix(self):
        if self.distance_matrix is None:
   	        raise ValueError(
                'no distance matrix has been computed. Please o so prior to calling this function')
        else:
            return self.distance_matrix

    def get_dissimilarity_matrix(self):
        if self.dissimilarity_matrix is None:
            raise ValueError(
                'no disssimilarity matrix has been compute. Pleasse do so prior to calling this function')
        else:
            return self.dissimilarity_matrix


    def get_shortest_path_maatrix(self):
        if self.shortest_path_matrix is None:
            raise ValueError(
                'no shortest path maatrix has been computed. Pleasse do so prior to calling this function')
        else:
            return self.shortest_path_matrix

        




