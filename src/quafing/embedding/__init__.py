import os

from .mds_embedder import MDSEmbedder as mds

embedders ={
	'mds':mds
}

def retrieve_embedder(method,mdpdf_collection=None):
    """
    Return initialized  specified embedder

    :param mdpdf_colection: optional; MultiPdfCollection for which embedding iss to  be calcuulated
    :return embedder: instance of specified embedder 
    """	

    _check_embedding_method(method)
    embedder = embedders[method]
    return embedder(mdpdf_collection)

def _check_embedding_method(method):
    """
    check whether requested embedder is supported.

    :param method: str; key for specified embeder in embedders dictionary
    """
    if method not in embedders:
        raise NotImplementedError(
            f"Embedding algorithm {method} is unknown")	





