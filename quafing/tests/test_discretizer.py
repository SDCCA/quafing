import pytest

import quafing
from quafing.discretization.discretize import discretize

path='test_data/test_data.xlsx'
metadata_true,data_true=quafing.load(path)
processed_data = quafing.preprocessing.PreProcessor(data_true,metadata_true)
processed_data.select_columns(select_all=True)

def test_discretize():
    disc = discretize(processed_data._data, processed_data._colmetadata, method = 'BayesianBlocks')
    assert disc is not None
    disc = discretize(processed_data._data, processed_data._colmetadata, method = 'BayesianBlocks', cols=['b'])
    assert disc[0]['Disc'] is None
    assert disc[1]['Disc'] is not None

def test_method_specification():
    with pytest.raises(RuntimeError, match=r'discretize requires a method to be specified, but none was specified'):
        disc = discretize(processed_data._data, processed_data._colmetadata)
    with pytest.raises(NotImplementedError):
        disc = discretize(processed_data._data, processed_data._colmetadata, method = 'Bbl')

def test_perform_discretization():
    with pytest.raises(ValueError, match=r'specified column types not all present in data'):
        disc = discretize(processed_data._data, processed_data._colmetadata, method='BayesianBlocks',cols=['d'])
    with pytest.raises(ValueError, match=r'Column specification contains mixed types'):
        disc = discretize(processed_data._data, processed_data._colmetadata, method='BayesianBlocks',cols=[2,'d'], byType=False)


