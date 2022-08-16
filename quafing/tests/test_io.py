import pytest
import numpy as np
import quafing

def test_file_format_error():
    with pytest.raises(NotImplementedError):
        path='test_data/test_data.abc'
        metadata,data=quafing.load(path)    

def test_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        path='test_data/test2.xlsx'
        metadata,data=quafing.load(path)

def test_read():
    path='test_data/test_data.xlsx'
    metadata,data=quafing.load(path) 
    assert type(metadata) is dict
    assert len(metadata) is 3
    assert data.shape == (4,3)
