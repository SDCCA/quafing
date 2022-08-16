from re import A
import sys
from py import process
from pyparsing import PrecededBy
from zmq import PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE

sys.path.insert(0, '/mnt/c/Documents and Settings/PranavChandramouli/Documents/One Drive/OneDrive - Netherlands eScience Center/Projects/Social_Dynamics/quafing')
import pytest
import numpy as np
import quafing
import quafing.preprocessing
import quafing.multipdf
import quafing.multipdf.multipdf
from quafing.multipdf.multi_dimensional_pdf import MultiDimensionalPDF

path='test_data.xlsx'
metadata_true,data_true=quafing.load(path)
data = quafing.preprocessing.PreProcessor(data_true,metadata_true)
data.select_columns(select_all=True)

def test_check_type():
    with pytest.raises(NotImplementedError):
        quafing.multipdf.multipdf.create_multi_pdf('Factrized', data._data, data._colmetadata,calculate=True)

def test_basic_validation():
    with pytest.raises(RuntimeError, match=r'raw data input is not of type pandas DataFrame'):
        a = np.random.randn(4,3)
        quafing.multipdf.multipdf.create_multi_pdf('factorized', a, data._colmetadata,calculate=True)
    with pytest.raises(RuntimeError):
        quafing.multipdf.multipdf.create_multi_pdf('factorized', data._data, a,calculate=True)

def test_base_calculate_pdf():
    a = MultiDimensionalPDF()
    with pytest.raises(NotImplementedError):
        a.calculate_pdf()

def test_disc_value():
    quafing.multipdf.multipdf.create_multi_pdf('factorized', data._data, data._colmetadata,calculate=True)
    
