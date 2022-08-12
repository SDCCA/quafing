from quafing.embedding import retrieve_embedder

def get_embedding(method,mdpdf_collection=None, dimension=2,**kwargs)
    embedder = retrieve_embedder(method,mdpdf_collection=mdpdf_collection)
    embeding = embedder.embed(dimension=dimension,return_all=True)
    return embedding

def get_embedder(method,mdpdf_collection=None)
    embedder = retrieve_embedder()
    return embedder

def plot_embedding(embedding,mdpdf_collection):
	    raise NotImplementedError(
	    	'plot_embedding not yet implemented')
	    