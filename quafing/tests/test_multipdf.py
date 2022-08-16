import pytest
import numpy as np
import quafing
from quafing.multipdf.multipdf import create_multi_pdf
from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF

path='test_data.xlsx'
metadata_true,data_true=quafing.load(path)
data = quafing.preprocessing.PreProcessor(data_true,metadata_true)
data.select_columns(select_all=True)
data.set_cont_disc([], by_type=True, complement=True)

def test_check_type():
    with pytest.raises(NotImplementedError):
        create_multi_pdf('Factrized', data._data, data._colmetadata,calculate=True)

def test_basic_validation():
    with pytest.raises(RuntimeError, match=r'raw data input is not of type pandas DataFrame'):
        a = np.random.randn(4,3)
        create_multi_pdf('factorized', a, data._colmetadata,calculate=True)
    with pytest.raises(RuntimeError):
        create_multi_pdf('factorized', data._data, a,calculate=True)

def test_base_calculate_pdf():
    a = MultiDimensionalPDF()
    with pytest.raises(NotImplementedError):
        a.calculate_pdf()

def test_density_method_not_None():
    with pytest.raises(RuntimeError, match=r'no method for density estimation specified'):
        create_multi_pdf('factorized', data._data, data._colmetadata,method=None, calculate=True,discretization= None)

def test_density_method_is_str():
    create_multi_pdf('factorized', data._data, data._colmetadata,method='Discrete1D', calculate=True,discretization= None)
    assert data._colmetadata[0]['density_method'] is 'Discrete1D'

def test_density_method_is_dict():
    discretization = [{"ColNames":"Col2", "density_method":"Discrete1D"}]
    create_multi_pdf('factorized', data._data, data._colmetadata,method='Discrete1D', calculate=True,discretization= None)
    assert data._colmetadata[1]['density_method'] is 'Discrete1D'

def test_disc_value():
    create_multi_pdf('factorized', data._data, data._colmetadata,calculate=True, method='Discrete1D',discretization= None)
    assert data._colmetadata[0]['Disc'] is None

def test_mdpdf_output():
    mdpdf = create_multi_pdf('factorized', data._data, data._colmetadata,calculate=True, method='Discrete1D',discretization= None)
    assert mdpdf._pdf is not None
    assert mdpdf._pdf_meta is not None
