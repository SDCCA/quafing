import os

from .mds_embedder import MDSEmbedding as mds

embedders ={
	'mds':mds
}

def retrieve_embedder(method,mdpdf_collection=None):

    _check_embedding_method(method)
    embedder = embedders[method]
    return embedder(mdpdf_collection)

def _check_embedding_method(method):
    if method not in embedders:
        raise NotImplementedError(
            f"Embedding algorithm {method} is unknown")	





