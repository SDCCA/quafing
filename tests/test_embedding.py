import pytest
import numpy as np
import quafing
from quafing.embedding.embed import get_embedder

path='tests/test_data/test_data.xlsx'
metadata_true,data_true=quafing.load(path)
data = quafing.PreProcessor(data_true,metadata_true)
data.select_columns(select_all=True)
data.set_cont_disc([], by_type=True, complement=True)
data.set_density_method(method='Discrete1D',cols=['b','c'])
data.split_to_groups(0)
mdpdfcol = quafing.create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
embedder = get_embedder('mds',mdpdfcol)

def test_check_embedding_method():
    with pytest.raises(NotImplementedError):
        get_embedder('dds',mdpdfcol)

def test_embed():
    with pytest.raises(ValueError, match=r'no valid return type specified. One of return_all and return_stress must be True'):
        embedder.embed(dimension=2,return_all=False, return_stress=False)
    with pytest.raises(ValueError, match=r'dimension must be an integer'):
        embedder.embed(dimension=2.5,return_all=True, return_stress=False)
    with pytest.raises(ValueError, match=r'Dimension of MDS computation must be greater than 1'):
        embedder.embed(dimension=1,return_all=True, return_stress=False)
    
def test_distance_matrix_error():
    with pytest.raises(ValueError):
        embedder.embed(dimension=2,return_all=True)

def test_eval_stress_v_dimension():
    mdpdfcol.calculate_distance_matrix(method='hellinger',pwdist='rms')
    embedder.embed(dimension=2,return_all=True)
    stresses = embedder.eval_stress_v_dimension(plot=False)
    assert len(stresses) is 2