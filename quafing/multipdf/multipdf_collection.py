""" collection of multi-dimensional pdfs"""
import os
import warnings
import numpy as np
import networkx as nx
import community
import metis

from quafing.distance.information_distance import information_distance



class MultiPdfCollection(object):
    """
    Collection of multi-dimensional pdfs of uniform type over a uniform set of dimensions. E.g. multi-dimensional pdfs
    of different groups of respondants for a given quuestionaire 
    """ 

    def __init__(self, collection, labels, metadata, mdpdftype, validate_metadata=True):
        """
        initialize, set attributes, and validate

        :param collection: list of multi-dimenssional pdfs (mdpdf)
        :param labels: list of labels for mdpdfs
        :param metadata: (column) metadata for mdpdfs
        :param mdpdftype: type of mdpdf 
        :param validate_metadata: bool (default=True). perform validation on metadata
        """

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
        """
        Basic validation on input. Check for duplicate labels, mismatch between labels and data, inconsistent mdpdf types
        """

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
        """
        Extended validation on meta data. Is provided metadata consistent with that of constituent pdfs and uniform.  
        """

        if not all([mdpdf._colmetadata == self._metadata for mdpdf in self._collection]):
            raise RuntimeError(
            'mismatch in metadata. mpdf column metadata does not match reference.')

        if not all([mdpdf._colmetadata == self._collection[0]._colmetadata for mdpdf in self._collection]):
            raise RuntimeError(
            'mismatch between column metadata of mdpdfs in collection ')


    def _calculate_all_mdpdfs(self,*args,**kwargs):
        """
        Calculate densities for all mdpdf objects in collection

        :param args: positional arguments for calculate_pdf() method of mdpdf object
        :param kwargs: keyword arguments for calculate_pdf() method of mdpdf object
        """
        for mdpdf in self._collection:
            mdpdf.caculate_pdf(*args,**kwargs)

 	
    def calculate_distance_matrix(self,method=None,pwdist=None,dims=None, return_result=False,kwargs_list=None):
        """
        Calculate distance matrix (i.e matrix of pairwise distances) for collection member mpdfs.

        :param method: str or list of str, method to be passed to information distance calculaation. 
                       One of valid keys in quafing.distance.__init__ ID_measuress
        :param pwdist: str, aggregation method for piecewise distance calculation. one of valid keys is 
                       quafing.distance.base_disctance._get_pwdist_func() pwdistfuncs
        :param dims: optional, str or list of str. dimensionality of (piecewise) mdpdfs for which (piecewise) distance is being calculated, 1d or nd.
                     defaults to use of auto_dims if dims=None
        :param return_result: bool (default False), return matrix if true, else update distance_matrix and _distance_matrix_type attributes
        :param kwargs_list: list of keyword arguments dictionaires to to be passed to information distance calculaation
        :return dist_matrix: optional distance matrix (if not returned object attribute is updated) 
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

        :param dist_matrix: distance matrix (numpy square matrix, all entries positive)
        :param return_result: bool (default=False). return resultant matrix. Else updaate object attributes 

        :return new_mat: A matrix with shortest distances.

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
        """
        return distance matrix
        """
        if self.distance_matrix is None:
   	        raise ValueError(
                'no distance matrix has been computed. Please o so prior to calling this function')
        else:
            return self.distance_matrix

    def get_dissimilarity_matrix(self):
        """
        return dissimilarity matrix
        """
        if self.dissimilarity_matrix is None:
            raise ValueError(
                'no disssimilarity matrix has been compute. Pleasse do so prior to calling this function')
        else:
            return self.dissimilarity_matrix


    def get_shortest_path_matrix(self):
        """
        return shortest path matrix
        """
        if self.shortest_path_matrix is None:
            raise ValueError(
                'no shortest path matrix has been computed. Pleasse do so prior to calling this function')
        else:
            return self.shortest_path_matrix

        




