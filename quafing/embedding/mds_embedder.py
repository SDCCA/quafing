import os
from types import MappingProxyType

from sklearn import manifold
from quafing.embedding.base_embedder import Embedder


class MDSEmbedder(Embedder):

    def __init__(self,mdpdf_collection=None):
        super().__init__(mdpdf_collection=mdpdf_collection)

        self._mds_defaults = MappingProxyType({
            'n_components':None, 
            'metric':True, 
            'max_iter':3000, 
            'eps':1e-9, 
            'random_state':None, 
            'dissimilarity':'precomputed', 
            'n_jobs':1
            })

        self._embedding_method = "MDS"
        self._mds_pars = dict(self._mds_defaults)


    def set_embedding_parameters(self, default=False, **kwargs):
        if default:
            self._mds_pars = dict(self._mds_defaults)
        else:
            self._mds_pars.update(kwargs)

    def embed(self, dimension=2, seed=3,return_all=False,return_stress=False):
        if not return_all and not return_stress:
            raise ValueError(
                'no valid return type specified. One of return_all and return_stress must be True') 
        if not isinstance(dimension,int):
            raise ValueError(
                'dimension must be an integer')
        if not dimension > 1:
            raise ValueError(
                "Dimension of MDS computation must be greater than 1")

        rsseed = np.random.RandomState(seed=seed)
        self.set_embedding_parameters(default=False,n_components=dimension,random_state=rsseed)

        mds = manifold.MDS(**self._mds_pars)

        if self._mdpdfc is not None:
            dist_matrix = self._mdpdfcc.get_distance_matrix()
        elif self._dmatrix is not None:
            dist_matrix = self._dmatrix
        else:
            raise RuntimeError(
                'No distance maatrix available to embed')        

        fit = mds.fit(dist_matrix)
        embedding = fit.embedding_
        stress = fit.stress_

        auxinfo = {**self._mds_pars,**{'seed':seed,'stress':stress,"embedding_method":self._embedding_method}}

        if return_stress:
            return stress
        else:
            return {'embedding':embedding,'auxinfo':auxinfo}

        

    def eval_stress_v_dimension(self,seed=3,plot=False):
    	

        if self._mdpdfc is not None:
            dm = self._mdpdfcc.get_distance_matrix()
        elif self._dmatrix is not None:
            dm = self._dmatrix

        max_dim = dm.shape[0] - 1

        stresses = list()
        dims = list(range(2, max_dim + 1))
        for d in dims:
            stresses.append(self.embed(dimension=d, seed=seed, return_stress=True))

        if plot:
            plt.plot(dims, stresses)
            plt.xlabel("Dimension")
            plt.ylabel("Stress")
            plt.title("MDS computed stress as function of dimension")
            plt.show()	
            return stresses
        else:
            return stresses

