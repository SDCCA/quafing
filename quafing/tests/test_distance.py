from re import A
import sys
from py import process
from pyparsing import PrecededBy
from zmq import PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE
sys.path.insert(0, '/mnt/c/Documents and Settings/PranavChandramouli/Documents/One Drive/OneDrive - Netherlands eScience Center/Projects/Social_Dynamics/quafing')
import pytest
import numpy as np
import quafing
import quafing.distance
import quafing.distance.information_distance
from scipy.stats import norm

#Test Dimensionality other than 1d and nd
def test_wrong_dims_kl(): 
    with pytest.raises(RuntimeError):
        quafing.distance.kl_divergence_distance._choose_sym_kl_div_func(dim='3d')

def test_wrong_dims_cos(): 
    with pytest.raises(RuntimeError):
        quafing.distance.cosine_distance._choose_cosine_func(dim='3d')

def test_wrong_dims_hd(): 
    with pytest.raises(RuntimeError):
        quafing.distance.hellinger_distance._choose_hellinger_func(dim='3d')

#Test nd "NotImplementedError" until implemented - TO BE CHANGED AFTER IMPLEMENTATION
def test_sym_kl_div_nd():
    with pytest.raises(NotImplementedError):
        quafing.distance.kl_divergence_distance.sym_kl_div_nd()

def test_cosine_nd():
    with pytest.raises(NotImplementedError):
        quafing.distance.cosine_distance.cosine_nd(is_discrete=True)
    with pytest.raises(NotImplementedError):
        quafing.distance.cosine_distance.cosine_nd(is_discrete=False)

def test_hellinger_nd():
    with pytest.raises(NotImplementedError):
        quafing.distance.hellinger_distance.hellinger_nd(is_discrete=True)
    with pytest.raises(NotImplementedError):
        quafing.distance.hellinger_distance.hellinger_nd(is_discrete=False)

#Test discrete calculations in 1d
def test_discrete_kl_div_1d_isinf_with_2nd_smaller_pdf():
    with pytest.warns(RuntimeWarning):
        p1 = {1: 0.2, 2: 0.1, 3: 0.7}
        p2 = {1: 0.2, 2: 0.1, 4: 0.7}
        assert np.isposinf(quafing.distance.kl_divergence_distance.sym_kl_div_1d(p1, p2, is_discrete=True))

def test_discrete_kl_zero_if_equal():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(quafing.distance.kl_divergence_distance.sym_kl_div_1d(p1, p2, is_discrete=True),0.)

def test_discrete_cosine_1d():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(quafing.distance.cosine_distance.cosine_1d(p1, p2, is_discrete=True), 0.)

def test_discrete_hellinger_1d_is_zero_ifequal():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(quafing.distance.hellinger_distance.hellinger_1d(p1, p2, is_discrete=True), 0.)

def test_discrete_hellinger_is_symmetric():
    p1 = {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.4}
    p2 = {1: 0.2, 2: 0.1, 4: 0.7}
    assert np.isclose(quafing.distance.hellinger_distance.hellinger_1d(p1, p2, is_discrete=True), quafing.distance.hellinger_distance.hellinger_1d(p2, p1, is_discrete=True))

#Test continuous distance calculation in 1D
def test_continuous_kl_converges_to_analytic_result():
    s1 = 1.0
    s2 = 2.0
    m1 = 1
    m2 = 1
    p1 = norm.freeze(loc=m1, scale=s1).pdf
    p2 = norm.freeze(loc=m2, scale=s2).pdf
    kl_dist = quafing.distance.kl_divergence_distance.continuous_kl_div_1d(p1, p2, base=None)
    analytic = np.log(s2/s1) + (s1**2 + (m1-m2)**2) / (2 * s2**2) - 0.5
    # On my machine relative error around 1e-16
    assert np.isclose(kl_dist, analytic)

def test_continuous_hellinger_converges_to_analytic_result():
    s1 = 1.0
    s2 = 2.0
    m1 = 1
    m2 = 1
    p1 = norm.freeze(loc=m1, scale=s1).pdf
    p2 = norm.freeze(loc=m2, scale=s2).pdf
    hell_dist = quafing.distance.hellinger_distance.continuous_hellinger_1d(p1, p2)
    analytic = (1 - np.sqrt((2*s1*s2)/(s1**2+s2**2)) * np.exp(-0.25*((m1-m2)**2)/(s1**2+s2**2)))
    analytic = np.sqrt(2 * analytic)
    # On my machine relative error around 1e-16
    assert np.isclose(hell_dist, analytic)

def test_get_ID_measure():
    p1 = {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.4}
    p2 = {1: 0.2, 2: 0.1, 4: 0.7}    
    with pytest.raises(RuntimeError, match=r'specification of information distance measure required'):
        quafing.distance.information_distance.information_distance(p1,p2,method=None)
    with pytest.raises(RuntimeError, match=r'specifiation of piecewise distance aggregation function required'):
        quafing.distance.information_distance.information_distance(p1,p2,method='kl',pwdist=None)

def test_check_distance_measure_method():
    with pytest.raises(NotImplementedError):
        p1 = {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.4}
        p2 = {1: 0.2, 2: 0.1, 4: 0.7}
        quafing.distance.information_distance.information_distance(p1,p2,method='None',pwdist='avg')

def test_validate_input():
    with pytest.raises(RuntimeError):
        p1 = {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.4}
        p2 = list({1: 0.2, 2: 0.1, 4: 0.7})
        quafing.distance.information_distance.information_distance(p1,p2,method='kl',pwdist='avg')