import pytest
import numpy as np
import pandas as pd
import quafing

path='tests/test_data/test_data.xlsx'
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
            "ColNames": ['col1', 'col2'],
            "QuestionNumbers": ['1', '2' ]
        }
        data_test = quafing.preprocessing.PreProcessor(data,metadata) 
    with pytest.raises(ValueError, match = r'Mismatch in length between data and metadata'):
        metadata = {
            "ColTypes": ['a'], 
            "ColNames": ['col1'],
            "QuestionNumbers": ['1']
        }
        data_test = quafing.preprocessing.PreProcessor(data,metadata)

def test_col_metadata():
    assert processed_data._rawcolmetadata is not None

def test_select_col():
    processed_data.select_columns(["Col1"],by_type=False)
    assert processed_data._data.shape == (4,1)
    processed_data.select_columns(["a","b"])
    assert processed_data._data.shape == (4,2)
    with pytest.warns(UserWarning):
        processed_data.select_columns(["col1"],by_type=False)
    with pytest.warns(UserWarning):
        processed_data.select_columns(["col1"])        

def test_shuffle():
    processed_data.select_columns(select_all=True)
    data2 = processed_data._data.copy()
    processed_data.shuffle()
    data = processed_data._data
    assert list(data.index) != list(data2.index) 

def test_randomize_data():
    processed_data.select_columns(select_all=True)
    data2 = processed_data._data.copy()
    processed_data.randomize_data('Col1')
    data = processed_data._data
    assert list(data['Col1']) != list(data2['Col1']) 

def test_set_cont_disc():
    processed_data.select_columns(select_all=True)
    processed_data.set_cont_disc(['a'], by_type=True)
    assert processed_data._colmetadata[0]['discrete'] is False
    assert processed_data._colmetadata[1]['discrete'] is True
    assert len([d['discrete'] for d in processed_data._colmetadata]) == len([d['ColTypes'] for d in processed_data._colmetadata])
    processed_data.set_cont_disc([1], by_type=False)
    assert processed_data._colmetadata[1]['discrete'] is False
    processed_data.set_cont_disc(['a'], by_type=True, complement=False, disccols=['b'])
    assert processed_data._colmetadata[2]['discrete'] is None
    assert processed_data._colmetadata[1]['discrete'] is True

def test_validate_by_label():
    with pytest.raises(ValueError, match=r'Column specification contains mixed types'):
        processed_data.select_columns(select_all=True)
        processed_data.set_cont_disc(cols=['Col1',2], by_type=False)
        
def test_set_density_method():
    processed_data.set_density_method(method='Discrete1D',cols=['b','c'])
    assert processed_data._colmetadata[0]['density_method'] is None
    assert processed_data._colmetadata[1]['density_method'] == 'Discrete1D' 
    processed_data.set_density_method(method=['Discrete1D', 'Discrete1D'],cols=['b','c'])
    assert processed_data._colmetadata[0]['density_method'] is None
    assert processed_data._colmetadata[1]['density_method'] == 'Discrete1D' 
    processed_data.set_density_method(method={'b': 'Discrete1D', 'c': 'Discrete1D'})
    assert processed_data._colmetadata[0]['density_method'] is None
    assert processed_data._colmetadata[1]['density_method'] == 'Discrete1D' 
    with pytest.raises(RuntimeError, match=r'no method for density estimation specified'):
        processed_data.set_density_method()

def test_group():
    processed_data.select_columns(select_all=True)
    data2 = processed_data._data.copy()
    processed_data._data.loc[2,'Col1'] = processed_data._data['Col1'][0]
    processed_data.group('Col1')
    data = processed_data._data
    assert list(data['Col1']) != list(data2['Col1']) 

def test_split_to_group():
    processed_data.select_columns(select_all=True)
    processed_data.split_to_groups('Col1')
    assert processed_data._groups is not None

def test_get_joint_discretization():
    disc = pd.DataFrame(None)
    disc = processed_data.get_joint_discretization(method = 'BayesianBlocks',return_result=True)
    assert disc is not None