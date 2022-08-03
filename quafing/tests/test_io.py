import sys
sys.path.insert(0, '/mnt/c/Documents and Settings/PranavChandramouli/Documents/One Drive/OneDrive - Netherlands eScience Center/Projects/Social_Dynamics/quafing')

import pytest

import numpy as np
import quafing

import quafing.io 

def test_file_format_error():
    with pytest.raises(NotImplementedError):
        path='test_data.abc'
        metadata,data=quafing.load(path)    

def test_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        path='test2.xlsx'
        metadata,data=quafing.load(path)

def test_read():
    path='test_data.xlsx'
    metadata,data=quafing.load(path) 
    assert type(metadata) is dict
    assert len(metadata) is 3
    assert data.shape == (4,3)
