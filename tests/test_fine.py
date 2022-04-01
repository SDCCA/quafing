import numpy as np

from scipy.stats import norm

from fine.fine import discrete_kl, continuous_kl, discrete_hellinger, \
    continuous_hellinger, discrete_cos


# Tests on all distance functions

def test_discrete_kl_infinite_when_support_of_2nd_pdf_smaller_than_first():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 4: 0.7}
    assert np.isposinf(discrete_kl(p1, p2))


def test_discrete_kl_zero_if_equal():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(discrete_kl(p1, p2), 0.)


def test_discrete_hellinger_zero_if_equal():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(discrete_hellinger(p1, p2), 0.)


def test_discrete_hellinger_is_symmetric():
    p1 = {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.4}
    p2 = {1: 0.2, 2: 0.1, 4: 0.7}
    assert np.isclose(discrete_hellinger(p1, p2), discrete_hellinger(p2, p1))


def test_discrete_cos_zero_if_equal():
    p1 = {1: 0.2, 2: 0.1, 3: 0.7}
    p2 = {1: 0.2, 2: 0.1, 3: 0.7}
    assert np.isclose(discrete_cos(p1, p2), 0.)


def test_discrete_cos_with_analytic_result():
    p1 = {1: 1.}
    p2 = {2: 1.}
    cos_dist = discrete_cos(p1, p2)
    analytic = np.pi/2
    assert np.isclose(cos_dist, analytic)


def test_continuous_kl_converges_to_analytic_result():
    s1 = 1.0
    s2 = 2.0
    m1 = 1
    m2 = 1
    p1 = norm.freeze(loc=m1, scale=s1).pdf
    p2 = norm.freeze(loc=m2, scale=s2).pdf
    kl_dist = continuous_kl(p1, p2, base=None)
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
    hell_dist = continuous_hellinger(p1, p2)
    analytic = (1 - np.sqrt((2*s1*s2)/(s1**2+s2**2)) * np.exp(-0.25*((m1-m2)**2)/(s1**2+s2**2)))
    analytic = np.sqrt(2 * analytic)
    # On my machine relative error around 1e-16
    assert np.isclose(hell_dist, analytic)
