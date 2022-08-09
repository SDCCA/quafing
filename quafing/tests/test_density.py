from re import A
import sys
from py import process
from pyparsing import PrecededBy
from zmq import PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE
sys.path.insert(0, '/mnt/c/Documents and Settings/PranavChandramouli/Documents/One Drive/OneDrive - Netherlands eScience Center/Projects/Social_Dynamics/quafing')
import pytest
import numpy as np
import pandas as pd
import quafing
import quafing.preprocessing
import quafing.density
from quafing.density.estimate_density import get_density_estimate

path='test_data.xlsx'
metadata_true,data_true=quafing.load(path)
processed_data = quafing.preprocessing.PreProcessor(data_true,metadata_true)
processed_data.select_columns(select_all=True)
processed_data.set_cont_disc(['a'], by_type=True, complement=False, disccols=['b'])
processed_data.set_density_method(method='Discrete1D',cols=['b','c'])

def test_get_density_estimate():
    density = get_density_estimate(processed_data._data['Col2'], method='Discrete1D')
    assert density is not None

def test_check_density_method():
    with pytest.raises(NotImplementedError):
        density = get_density_estimate(processed_data._data['Col2'], method='Discrete2D')

def test_check_discretization_info():
    with pytest.raises(RuntimeError, match=r'discretization schemes for intrinsically discrete data are not supported'):
        density = get_density_estimate(processed_data._data['Col2'], method='Discrete1D',discretization='2D')
    with pytest.raises(RuntimeError, match=r'Estimation of discrete densities for non-discrete data requires a user supplied discretization'):
        density = get_density_estimate(processed_data._data['Col1'], method='Discrete1D',discrete=processed_data._colmetadata[0]['discrete'])

def test_check_input_1d():
    with pytest.raises(RuntimeError, match=r'input data is not one dimensional.'):
        a = np.random.randn(4,3)
        density = get_density_estimate(a, method='Discrete1D',discrete=True)
    with pytest.raises(TypeError):
        density = get_density_estimate(processed_data._data, method='Discrete1D',discrete=True)
    
