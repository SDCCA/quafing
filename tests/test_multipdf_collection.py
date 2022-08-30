import pytest
import numpy as np
import quafing
from quafing.multipdf.multipdf import create_mdpdf_collection
from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF

path='tests/test_data/test_data.xlsx'
metadata_true,data_true=quafing.load(path)
data = quafing.preprocessing.PreProcessor(data_true,metadata_true)
data.select_columns(select_all=True)
data.set_cont_disc([], by_type=True, complement=True)
data.set_density_method(method='Discrete1D',cols=['b','c'])
data.split_to_groups(0)

def test_duplicate_labels():
    with pytest.warns(UserWarning, match=r'Duplicate labels were passed '):
        grouplabels = data._grouplabels.copy()
        grouplabels[0] = 'b'
        create_mdpdf_collection('factorized',data._groups,grouplabels,data._groupcolmetadata)

def test_length_data_metadata():
    with pytest.raises(RuntimeError):
        grouplabels = data._grouplabels.copy()
        grouplabels = np.append(grouplabels, 'e')
        create_mdpdf_collection('factorized',data._groups,grouplabels,data._groupcolmetadata)

def test_get_distance_matrix():
    mdpdfcol = create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
    with pytest.raises(ValueError):
        mdpdfcol.get_distance_matrix()

def test_get_dissimilarity_matrix():
    mdpdfcol = create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
    with pytest.raises(ValueError):
        mdpdfcol.get_dissimilarity_matrix()

def test_get_shortest_path_matrix():
    mdpdfcol = create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
    with pytest.raises(ValueError):
        mdpdfcol.get_shortest_path_matrix()

def test_calculate_distance_matrix():
    mdpdfcol = create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
    mdpdfcol.calculate_distance_matrix(method='hellinger',pwdist='rms')
    assert mdpdfcol.distance_matrix is not None

def test_calculate_shortest_path_matrix():
    mdpdfcol = create_mdpdf_collection('factorized',data._groups,data._grouplabels,data._groupcolmetadata)
    mdpdfcol.calculate_distance_matrix(method='hellinger',pwdist='rms')
    mdpdfcol.calculate_shortest_path_matrix()
    assert mdpdfcol.shortest_path_matrix is not None
