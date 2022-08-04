import sys
from py import process
from pyparsing import PrecededBy
sys.path.insert(0, '/mnt/c/Documents and Settings/PranavChandramouli/Documents/One Drive/OneDrive - Netherlands eScience Center/Projects/Social_Dynamics/quafing')

import pytest

import numpy as np
import pandas as pd
import quafing

import quafing.preprocessing


path='test_data.xlsx'
metadata_true,data_true=quafing.load(path)
processed_data = quafing.preprocessing.PreProcessor(data_true,metadata_true)

def test_data_type():
    with pytest.raises(RuntimeError, match=r'raw data input is not of type pandas DataFrame'):
        data = [1,2,3]
        metadata = [1,2,3] 
        data_test = quafing.preprocessing.PreProcessor(data,metadata) 

def test_check_keys():
    with pytest.raises(ValueError):
        metadata=metadata_true.copy()
        data=data_true.copy()
        metadata.pop('ColNames')
        data_test = quafing.preprocessing.PreProcessor(data,metadata) 

def test_check_keys_warning():
    with pytest.warns(UserWarning, match=r"Warning: key 'QuestionNumbers' does not exist in metadata dictionary"):
        metadata=metadata_true.copy()
        data=data_true.copy()
        metadata.pop('QuestionNumbers')
        data_test = quafing.preprocessing.PreProcessor(data,metadata) 

def test_check_dimensions():
    with pytest.raises(ValueError, match=r'mismatch in length of metadata fields'):
        data = pd.DataFrame(np.random.randint(10, size=(3,3)))
        metadata = {
            "ColTypes": 'a', 
            "ColNames": ['col1', 'col2']
        }
        data_test = quafing.preprocessing.PreProcessor(data,metadata) 
    with pytest.raises(ValueError, match = r'Mismatch in length between data and metadata'):
        metadata = {
            "ColTypes": ['a'], 
            "ColNames": ['col1']
        }
        data_test = quafing.preprocessing.PreProcessor(data,metadata)

def test_col_metadata():
    assert processed_data._rawcolmetadata is not None

def test_select_col():
    processed_data.select_columns(["Col1"],by_type=False)
    assert processed_data._data.shape == (4,1)
    processed_data.select_columns(["a","b"])
    assert processed_data._data.shape == (4,2)

def test_shuffle():
    processed_data.select_columns([],select_all=True)
    data = processed_data._data
    processed_data.shuffle()
    data2 = processed_data._data
    assert list(data1.index) != list(data2.index) 

def test_randomize_data():
    processed_data.select_columns([],select_all=True)
    data = processed_data._data
    data2 = data.copy()
    processed_data.randomize_data('Col1')
    data = processed_data._data
    assert list(data1.index) != list(data2.index) 
