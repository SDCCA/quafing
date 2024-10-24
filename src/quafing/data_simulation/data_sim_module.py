#!/usr/bin/env python3
# encoding: utf-8

"""
Data simulation module
----------------------

This module combines the scripts simulate_data.py, hypersphere.py, and spherical.py
to create a single module of all data simulation related algorithms.

 - Netherlands eScience Center
"""



# Standard imports for simulate_data
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

import simplejson

from fine import *

# Imports for hypersphere.py
import pickle
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pprint import pprint

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import array, cos, linspace, ones, pi, sin, zeros
from scipy.integrate import quad
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


import multiprocess
#from helpers import get_pca
import mca


"""
Simulate data for sparse FINE
-----------------------------

This script simulates data for Subak analysis which should enable me to verify
different statistical properties of the results, test how sparsity affects
FINE, etc...

Usage
------
simulate_date(N_params, N_questions, N_answers, N_groups, N_answers_per_group)

# N_params - Number of parameters in the model
# N_questions - Number of questions in the questionnaire
# N_answers - Number of answers per question
# N_groups - Number of groups (Subaks)
# N_answers_per_group -  Number of of answers per group (fixed)

Written by
----------

Omri Har-Shemesh,
University of Amsterdam

"""

def get_parameter_sets(N_params, N_groups):
    """ Returns N_groups sets of parameters randomly drawn. Each set of
    parameters will be linked to one group and will form the basis for
    distinguishing the groups.

    Args:
        N_params (int): Number of parameters per set
        N_groups (int): Number of groups.

    Returns: A list of parameter lists

    """
    return np.random.uniform(size=(N_groups, N_params))

def gs(X, row_vecs=True, norm = True):
    """ Performs the Gram-Schmidt orthonormalization of vectors given
    as row or column matrix X. Taken from comment by "ingmarschuster" in:
    https://gist.github.com/iizukak/1287876

    Args:
        X (ndarray): row or column matrix of vectors.

    Kwargs:
        row_vecs (bool, True): Are the vectors rows or columns?
        norm (bool, True): Should the vectors be normalized?

    """
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def get_random_vectors(N_vectors, dim):
    """ Constructs N_vectors random orthonormal vectors of dimension
    N_answers ** N_questions. These vectors will be used to generate the
    statistical model from which answers will be drawn.

    Args:
        N_vectors (int): Number of vectors to generate.
        dim (int): Dimension of the vectors to generate.

    Returns: An ndarray which is N_vectors x dim.

    """
    vs = np.random.uniform(size=(N_vectors, dim))
    return gs(vs)

def get_answers(N_questions, N_answers, probabilities, how_many):
    """ Returns simulated answers according to the probabilities joint PDF.
    This would represent the answers given by one 'Subak' or, more generally,
    by one group corresponding to a set of parameters.

    Args:
        N_questions (int): How many questions in the questionnaire?
        N_answers (int): How many answers per question are there?
        probabilities (ndarray N_answers x N_questions dimensions): Joint PDF.
        how_many (int): Number of simulations of answers to return.

    Returns: A list of lists of answers.

    """
    p_flat = probabilities.flatten()
    ans = np.random.choice(len(p_flat), size=how_many, p=p_flat)
    index = np.arange(len(p_flat)).reshape([N_answers] * N_questions)
    answers = []
    for a in ans:
        pos = np.where(index == a)
        pos = [k[0] for k in pos]
        answers.append(pos)
    return answers

def generate_models(N_questions, N_answers, N_params, N_groups):
    """ Generates statistical models for all N_groups with N_params, for
    N_questions with each N_answers.

    Args:
        N_questions (int): Number of questions in the questionnaire
        N_answers (int): Number of answers per question (fixed across questions)
        N_params (int): Number of parameters of the model.
        N_groups (int): Number of groups with different parameters.

    Returns: Probabilities for each answer in each question for each group.

    """
    # Number of possible answers to whole questionnaire:
    dim = N_answers ** N_questions
    vs = get_random_vectors(N_params, dim)
    params = get_parameter_sets(N_params, N_groups)
    group_vectors = params.dot(vs)
    norms = np.linalg.norm(group_vectors, axis=1)
    group_vectors = (group_vectors.T / norms).T
    probabilities = group_vectors ** 2
    return probabilities, params

def create_excel_file(filename, N_questions=12, N_answers=4, N_params=6, N_groups=20, N_answers_per_group=25, seed=1):
    """ Generates an excel file that fine.py can read and process. It also
        generates a json file with the parameters used to crate the excel
        file.

    Args:
        filename (str): The name of the file to create, without extension!

    Kwargs:
        N_questions (int, 12): Number of questions in the questionnaire
        N_answers (int, 4): Number of answers per question
        N_groups (int, 20): Number of groups ('Subaks')
        N_params (int, 25): Number of parameters of the model.

    Returns: Nothing

    """
    # Save the metadata to a json file.
    metadata = {
        'N_questions' : N_questions,
        'N_answers' : N_answers,
        'N_params' : N_params,
        'N_groups' : N_groups,
        'N_answers_per_group' : N_answers_per_group,
        'seed' : seed
    }

    column_types = ['o'] * N_questions
    column_types.insert(0, 'g')

    data = [column_types]
    questions = list(range(N_questions))
    questions.insert(0, 'name')
    data.append(questions)

    np.random.seed(seed)

    probs, params = generate_models(N_questions, N_answers, N_params, N_groups)
    for g in range(N_groups):
        ans = get_answers(N_questions, N_answers, probs[g,:], N_answers_per_group)
        g_name = chr(ord('a') + g % 26) + str(g // 26)
        for a in ans:
            a.insert(0, g_name)
            data.append(a)

    fn = open(filename + ".json", "w")
    metadata['params'] = params.tolist()
    simplejson.dump(metadata, fn, indent=" " * 4)
    fn.close()

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename + ".xlsx")
    df.to_excel(writer, 'Sheet1', header=False, index=False)
    writer.save()


def get_pca(df, columns, dim=2, align=True, permute=False, onehot=True):
    """ Computes a PCA from the given DataFrame and columns. Returns a list of
        Subak coordinates and list of Subak labels, similar to get_mds.
        The coordinates of each Subak are computed as the "mean individual",
        i.e. the Subak as a whole resides at the barycentric point of all
        farmers from this Subak.    
    Args:
        df (DataFrame): The data (unstandardized)
        columns (list): A list of column names to use for the PCA. (Must be
                        more than one!)
    Kwargs:
        dim (int): Number of principal components to calculate
        align(Bool): Should the coordinates be rotated and aligned for Betuas?
    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    if permute:
        df.name_1 = np.random.permutation(df.name_1)

    if onehot:
        data_normed = OneHotEncoder(sparse=False).fit_transform(df[columns])
    else:
        data_normed = StandardScaler().fit_transform(df[columns])
    pca = PCA(n_components=dim).fit_transform(data_normed)
    subaks = sorted(df.name_1.unique())
    pca_coords = np.zeros(shape=(len(subaks), dim))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the PCA space
        ind = df[df.name_1 == s].index
        pca_coords[i,:] = np.mean(pca[ind], axis=0)

    if align:
        pca_coords = align_coords(pca_coords)
    return pca_coords, subaks

def align_coords(coords):
    """ Rotates and reflects the coordinate system such that:
        1. Betuas, Keramas appears on the negative x-axis.
        2. Selukat is in the positive y-axis.
        
    """
    if coords.shape[1] == 1:
        return coords

    if coords.shape[0] < 13:
        return coords
    angle = -np.arctan(coords[2][1]/coords[2][0])
    new_coords = rotate_coords(coords, angle)
    if new_coords[2][0] > 0:
        new_coords = rotate_coords(coords, angle+np.pi)

    if new_coords[13][1] < 0:
        new_coords[:,1] = -new_coords[:,1]
    
    return new_coords

def rotate_coords(coords, angle=0):
    """ Rotates a list of coordinate pairs by a common angle.
        Args:
            coords: 2d numpy array of coordinates, result of MDS.fit_transform
        Kwargs:
            angle (default=0): Angle (in radians) to rotate the coordinates
        Returns: the transformed coordinates
    """
    if coords.shape[1] == 2:
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    # elif coords.shape[1] == 3:
        # rot_mat = np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        return coords
    return coords.dot(rot_mat)


def simulate_data(N_params, N_questions, N_answers, N_groups, N_answers_per_group):
    fname = "one_parameter_with_kl"
    create_excel_file("sim_res/%s" % fname, N_questions=5, N_answers_per_group=30, N_groups=30, N_params=1, seed=100)
    f = FINE("sim_res/%s.xlsx" % fname, start_row=3)
    f.plot_stress(method="kl")
    f.plot_embedding(d=2, method="kl")
    f.plot_embedding(d=3, method="kl")

"""
Hyperspehere.py algorithms
"""

def get_mfa(df, columns, dim=2):
    """ Computes the FactorAnalysis from the data.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Returns: TODO

    """
    fa = FactorAnalysis(n_components=2).fit_transform(df[columns])
    subaks = sorted(df.name_1.unique())
    mfa_coords = np.zeros(shape=(len(subaks), dim))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the MFA space
        ind = df[df.name_1 == s].index
        mfa_coords[i,:] = np.mean(fa[ind], axis=0)

    return mfa_coords

def get_mca(df, columns, dim=2):
    """ Calculates the MCA for the simulated questionnaire.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        dim (TODO): TODO

    Returns: TODO

    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    data = df.copy()

    MCA = mca(data, cols=list(columns), benzecri=True).fs_r(N=dim)
    if MCA.shape[1] < dim:
        raise Exception("MCA will only return values up to dimension {}".format(MCA.shape[1]))
    names = sorted(df.name_1.unique())
    mca_coords = np.zeros(shape=(len(names), dim))
    for i, s in enumerate(names):
        # Compute the position of the Subak in the MCA space
        ind = df[df.name_1 == s].index
        mca_coords[i,:] = np.mean(MCA[ind], axis=0)

    return mca_coords, names

def get_kernel_pca(df, columns, dim=2, kernel='rbf', onehot=False):
    """ Computes a Kernel PCA on the samples, to test how this performs in
    comparison with FINE.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        dim (TODO): TODO

    Returns: TODO

    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    if onehot:
        data = OneHotEncoder(sparse=False).fit_transform(df['columns'])
    else:
        data = StandardScaler().fit_transform(df[columns])

    kpca = KernelPCA(n_components=dim, kernel='rbf').fit_transform(data)
    subaks = sorted(df.name_1.unique())
    pca_coords = np.zeros(shape=(len(subaks), dim))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the PCA space
        ind = df[df.name_1 == s].index
        pca_coords[i,:] = np.mean(kpca[ind], axis=0)

    return pca_coords

def get_t_sne(df, columns, dim=2, **kwargs):
    """ Computes a T-SNE on the samples, to test how this performs in
    comparison with FINE.

    """

    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1

    # data = StandardScaler().fit_transform(df[columns])
    data = OneHotEncoder(sparse=False).fit_transform(df[columns])

    tsne = TSNE(n_components=dim, **kwargs).fit_transform(data)
    subaks = sorted(df.name_1.unique())
    pca_coords = np.zeros(shape=(len(subaks), dim))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the PCA space
        ind = df[df.name_1 == s].index
        pca_coords[i, :] = np.mean(tsne[ind], axis=0)

    return pca_coords


def mds_wrapper(dist_mat, dim=2, return_stress=False):
    """ Runs the MDS algorithm and returns the coordinates. The reason for
    this wrapper is so that the parameters given to MDS (n_jobs, n_init, etc..)
    will be consistent across all runs of the algorithm.

    Args:
        dist_mat (ndarray): Distance matrix to use in the MDS computation.

    Kwargs:
        dim (int, 2): The dimensionality of the MDS.

        return_stress (bol, False): Should the stress of the embedding also be
            returned?

    Returns: The coordinates calculated using the MDS algorithm.

    """
    mds = MDS(n_components=dim, dissimilarity='precomputed', n_init=15,
            max_iter=1000, n_jobs=7).fit(dist_mat)

    if return_stress:
        return mds.embedding_, mds.stress_
    return mds.embedding_

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Taken from:
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy

    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. They must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. Setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def bartlett_sphericity_test(df, columns):
    """ Performs the Bartlett test of Sphericity which should be done
        before using PCA.
        See:
        WHEN IS A CORRELATION MATRIX APPROPRIATE FOR FACTOR ANALYSIS?
            Psychological Bulletin, 1974, Vol. 81, No. 6

    Args:
        df (DataFrame): Input data
        columns (list-like): Columns to take into account

    Returns: p-value of the Bartlett test.

    """
    data = df[columns]
    N, p = data.shape

    R = np.linalg.det(data.corr())
    x = - (N - 1 - (2 * p + 5) / 6) * np.log(R)
    ddf = p * (p - 1) / 2

    return 1 - stats.chi2.cdf(x=x, df=ddf)

def compute_mds(df, dim=2, compute_joint=False, columns=None, return_stress=False):
    """ Computes the MDS embedding of the questionnaire results based on
    Fisher information distance.

    Args:
        df (DataFrame): DF containing the questionnaire responses.

    Kwargs:
        dim (int, 2): The dimensionality of the returned MDS.

        compute_join (Bool, True): Should the MDS compute also the distances
            from the joint distribution (assuming independence)?

        columns (list-like, None): If not None, a list of columns to take into
            account in the computation.

        return_stress (Bool, False): If true, returns the calculated stress for
            the MDS as well

    Returns: ndarray of coordinates. If compute_join, returns both coordinates
             for mean and for joint MDS.

    """
    # Estimate probabilities from questionnaire
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    if columns is None:
        columns = df.columns.drop("name_1")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(df[c], prefix=c))

    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df.name_1).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df.name_1).sum()
    prob = sum_df.divide(response_count, "index")

    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)

    # Compute the joint PDF assuming independence
    question_dict = defaultdict(list)
    for c in sqrt_prob.columns:
        question_dict["_".join(c.split("_")[:-1])].append(c)

    # prods = list(product(*question_dict.values()))
    # for i in range(len(prods)):
        # joint_mult = pd.DataFrame(sqrt_prob[list(prods[i])].prod(axis=1),
                # columns=[i])
        # if i == 0:
            # joint = joint_mult
        # else:
            # joint.ix[:,i] = joint_mult

    # Get all pairs of Subaks (without return)
    all_pairs = combinations(sqrt_prob.index, 2)

    res = {}
    res_joint = {}
    for c1, c2 in all_pairs:
        s1 = sqrt_prob.loc[c1]
        s2 = sqrt_prob.loc[c2]

        # To compute the Fisher distance, we use
        # FI = arccos(sum(sqrt(p_i*q_i))) for each question separately.
        pq = s1 * s2

        # This gives the sum over the multiplication performed on a per
        # question basis.
        questions = pq.groupby(lambda x: "_".join(x.split("_")[:-1])).sum()
        # Fix when rounding errors cause it to be slightly larger than 1.0
        questions[questions > 1] = 1.0
        FIs = np.arccos(questions)
        FI = FIs.mean()

        # joints = (joint.ix[c1] * joint.ix[c2]).sum()
        # if joints > 1.0: joints = 1.0
        # FI_joint = np.arccos(joints)

        # Fast joint
        FI_joint = np.arccos(questions.prod())

        c1i = int(c1)
        c2i = int(c2)

        if c1i in res:
            res[c1i][c2i] = FI
            res_joint[c1i][c2i] = FI_joint
        else:
            res[c1i] = {c2i: FI, c1i: 0.0}
            res_joint[c1i] = {c2i: FI_joint, c1i: 0.0}
        if c2i in res:
            res[c2i][c1i] = FI
            res_joint[c2i][c1i] = FI_joint
        else:
            res[c2i] = {c1i: FI, c2i: 0.0}
            res_joint[c2i] = {c1i: FI_joint, c2i: 0.0}

    dist_mat = pd.DataFrame(res).sort_index().sort_index(axis=1)
    dist_mat_joint = pd.DataFrame(res_joint).sort_index().sort_index(axis=1)

    # Perform the MDS
    if return_stress:
        mds_coords, mds_stress = mds_wrapper(dist_mat, dim=dim,
                return_stress=True)
        mds_coords_joint, joint_stress = mds_wrapper(dist_mat_joint, dim=dim,
                return_stress=True)
    else:
        mds_coords = mds_wrapper(dist_mat, dim=dim)
        mds_coords_joint = mds_wrapper(dist_mat_joint, dim=dim)

    if compute_joint:
        if return_stress:
            return mds_coords, mds_stress, mds_coords_joint, joint_stress
        else:
            return mds_coords, mds_coords_joint
    if return_stress:
        return mds_coords, mds_stress
    return mds_coords

def corr_to_dist(true_dist, coords):
    """
    Computes the correlation between the distance matrix 'true_dist' and the
    distance matrix computed from the coordinates 'coords'.
    """
    dm1 = squareform(true_dist)
    dm2 = pdist(coords)

    # Return the correlation coefficient:
    return np.corrcoef(dm1, dm2)[0,1]

def corr_between_coords(coords1, coords2):
    """ Computes the correlation between two sets of coordinates via the
    distance matrix

    Args:
        coords1 (ndarray): First array of coordinates
        coords2 (ndarray): Second array of coordinates

    Returns: Correlation between the two coordinates

    """
    dm1 = pdist(coords1)
    dm2 = pdist(coords2)

    return np.corrcoef(dm1, dm2)[0, 1]

def draw_questions(sqrt_probs, number_q=3, number_a=3, count=25):
    """ Draw answers from the join probability defined on the sphere.
        Currently assumes number of questions and answers is 3

    Args:
        number_q (TODO): TODO
        number_a (TODO): TODO
        sqrt_probs (TODO): TODO

    Kwargs:
        count: Number of answers to draw

    Returns: A list of responses based on the sqrt_probs provided

    """
    assert len(sqrt_probs) == number_a ** number_q

    rands = np.random.rand(count)
    res = []
    probs = sqrt_probs ** 2
    probs = probs.cumsum()
    ans_dict = get_ans_dict(number_q, number_a)
    for i in range(count):
        if rands[i] == 1.0:
            num = number_a ** number_q - 1
        else:
            num = np.argwhere(probs - rands[i] > 0)[0][0]

        # Convert number to answers
        res.append(list(ans_dict[num]))

    return res

def get_coords(angles):
    """ Computes the coordinates of a point on a unit hypersphere whose spherical
    coordinates (angles) are given.

    Args:
        angles (array): An array of angles

    Returns: Coordinates of the point thus specified

    """
    dim = angles.shape[0] + 1
    count = angles.shape[1]
    coords = ones(shape=(dim, count))

    for i in range(dim - 1):
        coords[i] *= cos(angles[i])
        coords[i+1:dim] *= sin(angles[i])

    return coords

def get_ans_dict(number_q=3, number_a=3):
    """ Computes a dictionary that can convert a number between 0 and
        number_a^number_q - 1 and responses to a questionnaire.

    Kwargs:
        number_q (TODO): TODO
        number_a (TODO): TODO

    Returns: TODO

    """
    max_num = number_a ** number_q
    res = {}
    for i in range(max_num):
        ans = []
        num = i
        for j in range(number_q - 1):
            exp = number_a ** (number_q - 1 - j)
            ans.append(num // exp)
            num = num % exp
        ans.append(num)
        res[i] = ans
    return res

def get_questionnaires(probs, number_q=3, number_a=3, count_answers=25):
    """ Produces a dataframe with simulated questionnaires, one for each
    probability distribution ("Subak")

    Args:
        probs (TODO): TODO

    Kwargs:
        count_answers (TODO): TODO

    Returns: TODO

    """
    q = []
    for i in range(probs.shape[0]):
        ans = draw_questions(probs[i], count=count_answers, number_q=number_q,
                number_a=number_a)
        for a in ans:
            a.insert(0, i + 1)
            q.append(a)

    columns = ['name_1']
    for i in range(number_q):
        columns.append("%s_%d" % (chr(ord("A")+i), i+2))

    q = pd.DataFrame(q, columns=columns)
    q.name_1 = q.name_1.astype(str)
    return q

def get_curve_samples(number_q=3, number_a=3, count=1000, samples=20, m=2,
        inds=None, random=False, rep=1, return_t=False, sin_angle=0,
        sin_angle_2=None):
    """ Returns a list of PDFs as samples from a curve.

    Kwrgs:
        number_q (int, 3): Number of questions to simulate in the
            questionnaire.

        number_a (int, 3): Number of possible answers per question.

        count (int, 1000): Number of points along the curve to calculate.

        samples (int, 20): Number of points to draw uniformly from the count
            number of points.

        inds (list-like, None): If not None, a list of indices to take from
            the count number of samples computed, instead of uniformly using
            samples.

        random (Bool, False): If set to true, sample the samples randomly form
            the curve, rather than take them at set intervals.

        rep (int, 1): How many sets of samples to return, in case sampling
            randomly rather than at set intervals.

        return_t (Bool, False): If true, the function will also return the
            value of t for each of the samples. This is necessary if one
            wants to compute the distance along the curve (using the analytic
            expression for the Fisher information).

        sin_angle (int, 0): Which angle to set to be the sine squared of t.
            Each selection gives a different type of curve.


    Returns: A two dimensional array of coordinates

    """
    dim = number_a ** number_q
    angles = zeros(shape=(dim - 1, count))

    for i in range(0, dim-1):
        angles[i, :] = linspace(0, pi/2, count)

    x = linspace(0, 1, count)
    angles[sin_angle, :] = (pi/2) * sin(x * pi * m)**2

    coords = get_coords(angles)

    if inds != None:
        if return_t:
            ts = x[inds]
            return coords[:, inds].T, ts
        return coords[:, inds].T

    if random:
        if rep == 1:
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                return coords[:, inds].T, ts
            return coords[:, inds].T
        coords_arr = []
        for i in range(rep):
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                coords_arr.append([coords[:, inds].T, ts])
            coords_arr.append(coords[:, inds].T)
        return coords_arr

    if return_t:
        ts = x[::count//samples]
        return coords[:,::count//samples].T, ts
    return coords[:,::count//samples].T

def get_alternative_curve_samples(number_q=3, number_a=3, count=1000, samples=20, m=2,
        inds=None, random=False, rep=1, return_t=False, sin_angle=0):
    """ Returns a list of PDFs as samples from a curve.

    Kwrgs:
        number_q (int, 3): Number of questions to simulate in the
            questionnaire.

        number_a (int, 3): Number of possible answers per question.

        count (int, 1000): Number of points along the curve to calculate.

        samples (int, 20): Number of points to draw uniformly from the count
            number of points.

        inds (list-like, None): If not None, a list of indices to take from
            the count number of samples computed, instead of uniformly using
            samples.

        random (Bool, False): If set to true, sample the samples randomly form
            the curve, rather than take them at set intervals.

        rep (int, 1): How many sets of samples to return, in case sampling
            randomly rather than at set intervals.

        return_t (Bool, False): If true, the function will also return the
            value of t for each of the samples. This is necessary if one
            wants to compute the distance along the curve (using the analytic
            expression for the Fisher information).

        sin_angle (int, 0): Which angle to set to be the sine squared of t.
            Each selection gives a different type of curve.

    Returns: A two dimensional array of coordinates

    """
    dim = number_a ** number_q
    angles = zeros(shape=(dim - 1, count))

    for i in range(0, dim-1):
        angles[i, :] = linspace(3*pi/8, 4*pi/8, count)

    x = linspace(0, 1, count)
    angles[sin_angle, :] = (pi/2) * sin(x * pi * m)**2

    coords = get_coords(angles)

    if inds != None:
        if return_t:
            ts = x[inds]
            return coords[:, inds].T, ts
        return coords[:, inds].T

    if random:
        if rep == 1:
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                return coords[:, inds].T, ts
            return coords[:, inds].T
        coords_arr = []
        for i in range(rep):
            inds = np.random.choice(list(range(count)), size=samples, replace=False)
            inds = sorted(inds)
            if return_t:
                ts = x[inds]
                coords_arr.append([coords[:, inds].T, ts])
            coords_arr.append(coords[:, inds].T)
        return coords_arr

    if return_t:
        ts = x[::count//samples]
        return coords[:,::count//samples].T, ts
    return coords[:,::count//samples].T

def get_true_mds(probs, dim=2):
    """ Returns the mds embedding of the "true" distance matrix based on the
    complete joint pdf.

    Args:
        probs (TODO): TODO

    Returns: TODO

    """
    true_dist_mat = get_true_dist_mat(probs)
    true_mds = mds_wrapper(true_dist_mat, dim=dim)

    return true_mds

def get_true_dist_mat(probs):
    """ Computes the "ground truth" distance matrix from the probabilities
    themselves

    Args:
        probs (TODO): TODO

    Returns: TODO

    """
    samples = probs.shape[0]
    true_dist_mat = zeros(shape=(samples, samples))
    for i, j in combinations(range(samples), 2):
        ij_mult = (probs[i] * probs[j]).sum()
        true_dist_mat[i, j] = np.arccos(ij_mult)
        true_dist_mat[j, i] = true_dist_mat[i, j]

    return true_dist_mat

def g(t, m, dim):
    """ Computes the Fisher information at point t

    """
    angles = zeros(dim - 1)

    angles[0] = (pi/2) * sin(t * pi * m)**2
    for i in range(1, dim-1):
        angles[i] = (pi/2) * t

    sin2 = sin(angles) ** 2
    sin_sum = sin2[0]
    for i in range(1, dim - 1):
        sin_sum += sin_sum * sin2[i]

    res = 4 * (m * (pi**2) * sin(m * pi * t) * cos(m * pi * t)) ** 2
    res += (pi**2) * sin_sum
    return np.sqrt(res)

def get_curved_fisher_distance(t1, t2, m, dim):
    """ Computes the true fisher distance between two points along the curve
    based on analytically computing it.

    Args:
        t1 (TODO): TODO
        t2 (TODO): TODO
        m (TODO): TODO
        dim (TODO): TODO

    Returns: TODO

    """
    return quad(g, t1, t2, args=(m, dim))[0]

def align_pca_mds(pca, mds):
    """ Aligns the coordinates obtained by PCA and those obtained by MDS
        so that they appear the closest. This is done by solving the
        Procrostean problem.

    Args:
        pca (ndarray): Array of coordinates obtained from PCA
        mds (ndarray): Array of coordinates obtained by MDS.

    Returns: The MDS after scaling, rotating and possibly reflecting so that
        it will best fit the PCA result. This is done by using the solution
        to the Procrustean problem.

    """

    d, mds_new, tform = procrustes(pca, mds)
    return mds_new

def plot_fi_pca_vs_m():
    """ Runs the simulation for each m separately and computes both
    correlations with MDS and with PCA. Then plots all on the same graph.

    """
    # Define parameters for the simulation
    start = time.time()
    samples = 50
    ms = [1, 2, 3, 4, 5, 6, 7, 8]
    responses = 50
    number_q = 3
    number_a = 3
    dim = 2
    kappa = 0
    random = False  # Are the groups sampled randomly from the line or
                    # equidistant?
    rep = 100

    save_file = True
    srand = "_random" if random else ""
    with_tsne = True
    stsne = "_tsne" if with_tsne else ""
    filename = "../paper/images/corr_fi_mca/" + \
            "corr_vs_m_kappa_%d_samples_%d_responses_%d_nq_%d_na_%d_rep_%d%s%s"
    # (The one run without kappa in filename was kappa=0)

    filename = filename % (kappa, samples, responses, number_q, number_a, rep,
            srand, stsne)

    inds = None

    corr_mca = []
    corr_joint = []
    corr_pca = []
    corr_tsne = []
    for m in ms:
        print("m = %d" % m)
        # Simulate the curve and a questionnaire
        c_mca, c_pca = [], []
        c_joint = []
        c_tsne = []
        def compute_embeddings(i):
            probs = get_curve_samples(number_q=number_q, number_a=number_a, samples=samples, m=m, inds=inds, random=random, sin_angle=kappa)
            df = get_questionnaires(probs, count_answers=responses, number_q=number_q,
                    number_a=number_a)
            true_dist_mat = get_true_dist_mat(probs)
            true_mds = get_true_mds(probs)

            # Get Fisher map
            mds, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

            # Compute PCA of the questionnaire
            df.name_1 = df.name_1.astype(int)
            pca, subaks = get_pca(df, df.columns.drop("name_1"), dim=dim)
            mca, subaks = get_mca(df, df.columns.drop("name_1"), dim=dim)
            tsne = get_t_sne(df, df.columns.drop("name_1"), dim=dim)

            return corr_to_dist(true_dist_mat, mca),\
                    corr_to_dist(true_dist_mat, pca), \
                    corr_to_dist(true_dist_mat, mds_joint), \
                    corr_to_dist(true_dist_mat, tsne)

        p = multiprocess.Pool()
        res = p.map_async(compute_embeddings, range(rep)).get()

        # Compute correlations between true mds, fisher and pca

        c_mca = [r[0] for r in res]
        c_pca = [r[1] for r in res]
        c_joint = [r[2] for r in res]
        c_tsne = [r[3] for r in res]
        # c_pca.append(corr_to_dist(true_dist_mat, pca))
        # c_joint.append(corr_to_dist(true_dist_mat, mds_joint))

        corr_mca.append([np.mean(c_mca), np.std(c_mca), c_mca])
        corr_pca.append([np.mean(c_pca), np.std(c_pca), c_pca])
        corr_joint.append([np.mean(c_joint), np.std(c_joint), c_joint])
        corr_tsne.append([np.mean(c_tsne), np.std(c_tsne), c_tsne])

    corr_mca_mean = array([c[0] for c in corr_mca])
    corr_mca_5 = array([np.percentile(c[2], 5) for c in corr_mca])
    corr_mca_95 = array([np.percentile(c[2], 95) for c in corr_mca])
    corr_mca_std = array([c[1] for c in corr_mca])

    corr_pca_mean = array([c[0] for c in corr_pca])
    corr_pca_5 = array([np.percentile(c[2], 5) for c in corr_pca])
    corr_pca_95 = array([np.percentile(c[2], 95) for c in corr_pca])
    corr_pca_std = array([c[1] for c in corr_pca])

    corr_joint_mean = array([c[0] for c in corr_joint])
    corr_joint_5 = array([np.percentile(c[2], 5) for c in corr_joint])
    corr_joint_95 = array([np.percentile(c[2], 95) for c in corr_joint])
    corr_joint_std = array([c[1] for c in corr_joint])

    corr_tsne_mean = array([c[0] for c in corr_tsne])
    corr_tsne_5 = array([np.percentile(c[2], 5) for c in corr_tsne])
    corr_tsne_95 = array([np.percentile(c[2], 95) for c in corr_tsne])
    corr_tsne_std = array([c[1] for c in corr_tsne])

    plt.errorbar(ms, corr_joint_mean, yerr=(corr_joint_mean-corr_joint_5, corr_joint_95-corr_joint_mean), label="FINE")
    plt.errorbar(ms, corr_mca_mean, yerr=(corr_mca_mean-corr_mca_5, corr_mca_95-corr_mca_mean), label="MCA")
    plt.errorbar(ms, corr_pca_mean, yerr=(corr_pca_mean-corr_pca_5, corr_pca_95-corr_pca_mean), label="PCA")
    plt.errorbar(ms, corr_tsne_mean, yerr=(corr_tsne_mean-corr_tsne_5, corr_tsne_95-corr_tsne_mean), label="T-SNE")

    plt.title("Correlations between FI/MCA/PCA/T-SNE and True values")
    plt.xlabel(r"$m$")
    plt.ylabel("Correlation")
    plt.legend()
    print("Calculation took %.2f seconds" % (time.time()-start))
    if save_file:
        # Save the current plot
        plt.savefig(filename + ".pdf", bbox_inches="tight")

        # Save the data for the plot
        f = open(filename + ".pkl", "wb")
        pickle.dump(dict(
            corr_mca = corr_mca,
            corr_pca = corr_pca,
            corr_fi = corr_joint,
            corr_tsne = corr_tsne,
            ms = ms,
            samples = samples,
            responses = responses,
            nq = number_q,
            na = number_a,
            dim = dim,
            rep = rep
            ), f)
        f.close()

        print("Saved figure to %s.pdf" % filename)

    plt.show()

def plot_fi_pca_vs_resp():
    """ Runs the simulation for each m separately and computes both
    correlations with MDS and with PCA. Then plots all on the same graph.
    This computes them as function of the number of responses N.

    """
    # Define parameters for the simulation
    start = time.time()
    samples = 20
    m = 3
    resp_arr = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500, 1000]
    resp_arr = [10, 15, 20, 25, 50, 75, 100]
    number_q = 3
    number_a = 3
    dim = 2
    kappa = 0
    random = False  # Are the groups sampled randomly from the line or
                    # equidistant?
    rep = 25
    rep = 5
    rep = 100

    save_file = True
    srand = "_random" if random else ""
    filename = "../paper/images/corr_fi_mca/" + \
            "corr_vs_N_kappa_%d_samples_%d_m_%d_nq_%d_na_%d_rep_%d%s"
    # (The one run without kappa in filename was kappa=0)

    filename = filename % (kappa, samples, m, number_q, number_a, rep, srand)

    inds = None

    corr_mca = []
    corr_joint = []
    corr_pca = []
    for responses in resp_arr:
        print("N = %d" % responses)
        # Simulate the curve and a questionnaire
        c_mca, c_pca = [], []
        c_joint = []
        def compute_embeddings(i):
            probs = get_curve_samples(number_q=number_q, number_a=number_a, samples=samples, m=m, inds=inds, random=random, sin_angle=kappa)
            df = get_questionnaires(probs, count_answers=responses, number_q=number_q,
                    number_a=number_a)
            true_dist_mat = get_true_dist_mat(probs)
            true_mds = get_true_mds(probs)

            # Get Fisher map
            mds, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

            # Compute PCA of the questionnaire
            df.name_1 = df.name_1.astype(int)
            pca, subaks = get_pca(df, df.columns.drop("name_1"), dim=dim)
            mca, subaks = get_mca(df, df.columns.drop("name_1"), dim=dim)

            return corr_to_dist(true_dist_mat, mca), corr_to_dist(true_dist_mat, pca), corr_to_dist(true_dist_mat, mds_joint)

        p = multiprocess.Pool(7)
        res = p.map_async(compute_embeddings, range(rep)).get()

        # Compute correlations between true mds, fisher and pca

        c_mca = [r[0] for r in res]
        c_pca = [r[1] for r in res]
        c_joint = [r[2] for r in res]
        # c_pca.append(corr_to_dist(true_dist_mat, pca))
        # c_joint.append(corr_to_dist(true_dist_mat, mds_joint))

        corr_mca.append([np.mean(c_mca), np.std(c_mca), c_mca])
        corr_pca.append([np.mean(c_pca), np.std(c_pca), c_pca])
        corr_joint.append([np.mean(c_joint), np.std(c_joint), c_joint])

    corr_mca_mean = array([c[0] for c in corr_mca])
    corr_mca_5 = array([np.percentile(c[2], 5) for c in corr_mca])
    corr_mca_95 = array([np.percentile(c[2], 95) for c in corr_mca])
    corr_mca_std = array([c[1] for c in corr_mca])

    corr_pca_mean = array([c[0] for c in corr_pca])
    corr_pca_5 = array([np.percentile(c[2], 5) for c in corr_pca])
    corr_pca_95 = array([np.percentile(c[2], 95) for c in corr_pca])
    corr_pca_std = array([c[1] for c in corr_pca])

    corr_joint_mean = array([c[0] for c in corr_joint])
    corr_joint_5 = array([np.percentile(c[2], 5) for c in corr_joint])
    corr_joint_95 = array([np.percentile(c[2], 95) for c in corr_joint])
    corr_joint_std = array([c[1] for c in corr_joint])

    plt.errorbar(resp_arr, corr_joint_mean, yerr=(corr_joint_mean-corr_joint_5, corr_joint_95-corr_joint_mean), label="FINE", color="blue")
    plt.errorbar(resp_arr, corr_mca_mean, yerr=(corr_mca_mean-corr_mca_5, corr_mca_95-corr_mca_mean), label="MCA", color="green")
    plt.errorbar(resp_arr, corr_pca_mean, yerr=(corr_pca_mean-corr_pca_5, corr_pca_95-corr_pca_mean), label="PCA", color="red")

    plt.title("Correlations between FI/MCA/PCA and True values")
    plt.xlabel(r"Responses per group")
    plt.ylabel("Correlation")
    plt.legend()
    print("Calculation took %.2f seconds" % (time.time()-start))
    if save_file:
        # Save the current plot
        plt.savefig(filename + ".pdf", bbox_inches="tight")

        # Save the data for the plot
        f = open(filename + ".pkl", "wb")
        pickle.dump(dict(
            corr_mca = corr_mca,
            corr_pca = corr_pca,
            corr_fi = corr_joint,
            m = m,
            samples = samples,
            resp_arr = resp_arr,
            nq = number_q,
            na = number_a,
            dim = dim,
            rep = rep
            ), f)
        f.close()

        print("Saved figure to %s.pdf" % filename)

    plt.show()

def plot_fi_pca_vs_samples():
    """ Runs the simulation for each m separately and computes both
    correlations with MDS and with PCA. Then plots all on the same graph.
    This computes them as function of the number of the number of groups.

    """
    # Define parameters for the simulation
    start = time.time()
    samples_arr = [15, 20, 30, 50, 75, 100]
    samples_arr = [10, 20, 30, 40, 50]
    m = 3
    resp_arr = [10, 15, 20, 25, 50, 75, 100]
    responses = 50
    number_q = 3
    number_a = 3
    dim = 2
    kappa = 1
    random = False  # Are the groups sampled randomly from the line or
                    # equidistant?
    rep = 100

    save_file = True
    srand = "_random" if random else ""
    filename = "../paper/images/corr_fi_mca/" + \
            "corr_vs_samples_kappa_%d_responses_%d_m_%d_nq_%d_na_%d_rep_%d%s"
    # (The one run without kappa in filename was kappa=0)

    filename = filename % (kappa, responses, m, number_q, number_a, rep, srand)

    inds = None

    corr_mca = []
    corr_joint = []
    corr_pca = []
    for samples in samples_arr:
        print("%d samples" % samples)
        # Simulate the curve and a questionnaire
        c_mca, c_pca = [], []
        c_joint = []
        def compute_embeddings(i):
            probs = get_curve_samples(number_q=number_q, number_a=number_a,
                    samples=samples, m=m, inds=inds, random=random,
                    sin_angle=kappa)
            df = get_questionnaires(probs, count_answers=responses, number_q=number_q,
                    number_a=number_a)
            true_dist_mat = get_true_dist_mat(probs)
            true_mds = get_true_mds(probs)

            # Get Fisher map
            mds, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

            # Compute PCA of the questionnaire
            df.name_1 = df.name_1.astype(int)
            pca, subaks = get_pca(df, df.columns.drop("name_1"), dim=dim)
            mca, subaks = get_mca(df, df.columns.drop("name_1"), dim=dim)

            return corr_to_dist(true_dist_mat, mca), \
                corr_to_dist(true_dist_mat, pca), corr_to_dist(true_dist_mat,
                        mds_joint)

        p = multiprocess.Pool(7)
        res = p.map_async(compute_embeddings, range(rep)).get()

        # Compute correlations between true mds, fisher and pca

        c_mca = [r[0] for r in res]
        c_pca = [r[1] for r in res]
        c_joint = [r[2] for r in res]
        # c_pca.append(corr_to_dist(true_dist_mat, pca))
        # c_joint.append(corr_to_dist(true_dist_mat, mds_joint))

        corr_mca.append([np.mean(c_mca), np.std(c_mca), c_mca])
        corr_pca.append([np.mean(c_pca), np.std(c_pca), c_pca])
        corr_joint.append([np.mean(c_joint), np.std(c_joint), c_joint])

    corr_mca_mean = array([c[0] for c in corr_mca])
    corr_mca_5 = array([np.percentile(c[2], 5) for c in corr_mca])
    corr_mca_95 = array([np.percentile(c[2], 95) for c in corr_mca])
    corr_mca_std = array([c[1] for c in corr_mca])

    corr_pca_mean = array([c[0] for c in corr_pca])
    corr_pca_5 = array([np.percentile(c[2], 5) for c in corr_pca])
    corr_pca_95 = array([np.percentile(c[2], 95) for c in corr_pca])
    corr_pca_std = array([c[1] for c in corr_pca])

    corr_joint_mean = array([c[0] for c in corr_joint])
    corr_joint_5 = array([np.percentile(c[2], 5) for c in corr_joint])
    corr_joint_95 = array([np.percentile(c[2], 95) for c in corr_joint])
    corr_joint_std = array([c[1] for c in corr_joint])

    plt.errorbar(samples_arr, corr_joint_mean, yerr=(corr_joint_mean-corr_joint_5, corr_joint_95-corr_joint_mean), label="FINE", color="blue")
    plt.errorbar(samples_arr, corr_mca_mean, yerr=(corr_mca_mean-corr_mca_5, corr_mca_95-corr_mca_mean), label="MCA", color="green")
    plt.errorbar(samples_arr, corr_pca_mean, yerr=(corr_pca_mean-corr_pca_5, corr_pca_95-corr_pca_mean), label="PCA", color="red")

    plt.title("Correlations between FI/MCA/PCA and True values")
    plt.xlabel(r"Number of groups")
    plt.ylabel("Correlation")
    plt.legend()
    print("Calculation took %.2f seconds" % (time.time()-start))
    if save_file:
        # Save the current plot
        plt.savefig(filename + ".pdf", bbox_inches="tight")

        # Save the data for the plot
        f = open(filename + ".pkl", "wb")
        pickle.dump(dict(
            corr_mca = corr_mca,
            corr_pca = corr_pca,
            corr_fi = corr_joint,
            m = m,
            samples_arr = samples_arr,
            responses = responses,
            nq = number_q,
            na = number_a,
            dim = dim,
            rep = rep
            ), f)
        f.close()

        print("Saved figure to %s.pdf" % filename)

    plt.show()

def plot_fi_pca_vs_m_random_samples():
    """ Plots the same as above, but sampling randomly uniformly from the
    curve rather than taking at set intervals.

    """
    # Define parameters for the simulation
    start = time.time()
    samples = 100
    ms = [1, 2, 3, 4, 5, 6, 7, 8]
    responses = 1000
    number_q = 5
    number_a = 4
    dim = 2
    rep = 10

    corr_fi = []
    corr_pca = []
    corr_both = []
    for m in ms:
        print("m = %d" % m)
        # Simulate the curve and a questionnaire
        c_fi, c_pca = [], []
        c_both = []
        probs = get_curve_samples(number_q=number_q, number_a=number_a, samples=samples, m=m, inds=None,
                random=True, rep=rep)
        if rep == 1:
            probs = [probs]
        for i in range(rep):
            df = get_questionnaires(probs[i], count_answers=responses, number_q=number_q,
                    number_a=number_a)
            true_dist_mat = get_true_dist_mat(probs[i])
            true_mds = get_true_mds(probs[i])

            # Get Fisher map
            mds = compute_mds(df, dim=dim)

            # Compute PCA of the questionnaire
            df.name_1 = df.name_1.astype(int)
            pca, subaks = get_pca(df, df.columns.drop("name_1"), dim=dim,
                    align=False)

            # Compute correlations between true mds, fisher and pca
            c_fi.append(corr_to_dist(true_dist_mat, mds))
            c_pca.append(corr_to_dist(true_dist_mat, pca))
            c_both.append(corr_between_coords(pca, mds))
        corr_fi.append([np.mean(c_fi), np.std(c_fi), c_fi])
        corr_pca.append([np.mean(c_pca), np.std(c_pca), c_pca])
        corr_both.append([np.mean(c_both), np.std(c_both), c_both])

    corr_fi_mean = array([c[0] for c in corr_fi])
    corr_fi_5 = array([np.percentile(c[2], 5) for c in corr_fi])
    corr_fi_95 = array([np.percentile(c[2], 95) for c in corr_fi])

    corr_pca_mean = [c[0] for c in corr_pca]
    corr_pca_std = [c[1] for c in corr_pca]
    corr_pca_5 = array([np.percentile(c[2], 5) for c in corr_pca])
    corr_pca_95 = array([np.percentile(c[2], 95) for c in corr_pca])

    corr_both_mean = [c[0] for c in corr_both]
    corr_both_std = [c[1] for c in corr_both]
    corr_both_5 = array([np.percentile(c[2], 5) for c in corr_both])
    corr_both_95 = array([np.percentile(c[2], 95) for c in corr_both])

    plt.errorbar(ms, corr_fi_mean, yerr=(corr_fi_mean - corr_fi_5, corr_fi_95-corr_fi_mean), label="FI Correlations", color="blue")
    plt.errorbar(ms, corr_pca_mean, yerr=(corr_pca_mean - corr_pca_5, corr_pca_95-corr_pca_mean), label="PCA Correlations", color="red")
    plt.errorbar(ms, corr_both_mean, yerr=(corr_both_mean - corr_both_5, corr_both_95-corr_both_mean), label="Correlation between PCA and FI", color="green")
    # plt.errorbar(ms, corr_pca_mean, yerr=corr_fi_std, label="PCA Correlations", color="red")
    plt.title("Correlations between FI/PCA and True values")
    plt.xlabel(r"$m$")
    plt.ylabel("Correlation")
    plt.legend()
    print("Calculation took %.2f seconds" % (time.time()-start))
    # plt.savefig("../writing/images/correlation_fi_pca_qs_%d_as_%d.pdf", bbox_inches="tight")
    plt.show()

def get_curved_mds():
    """ Computes the curved MDS from samples where t is known for each point
        along the curve.

    """
    samples, ts = get_curve_samples(return_t=True)

    dist_mat = zeros(shape=(len(ts), len(ts)))
    for i, j in combinations(list(range(len(ts))), 2):
        t1 = min([ts[i], ts[j]])
        t2 = max([ts[i], ts[j]])
        dist_mat[i, j] = get_curved_fisher_distance(t1, t2, m=2, dim=27)
        dist_mat[j, i] = dist_mat[i, j]

    mds_curved = mds_wrapper(dist_mat)

    mds_curved = mds_curved / mds_curved.max()

    mds_cosine = get_true_mds(samples)
    mds_cosine = mds_cosine / mds_cosine.max()

    plt.scatter(mds_curved[:, 0], mds_curved[:, 1], color='blue')
    plt.scatter(mds_cosine[:, 0], mds_cosine[:, 1], color='red')
    plt.show()

def plot_pca_fi():
    """ Plots one plot with both PCA and FI results after adjusting them
        to fit each other as best as possible.

    Returns: Nothing

    """
    samples = 50
    m = 3
    responses = 50
    number_q = 8
    number_a = 3
    dim = 2
    rep = 1
    kappa = 1
    save_file = True
    alternative = False
    show_pca = True
    show_kpca = True
    show_tsne = True
    salt = "alternative_" if alternative else ""
    spca = "_pca" if show_pca else ""
    skpca = "_kpca" if show_kpca else ""
    stsne = "_tsne" if show_tsne else ""
    filename = \
        "../paper/images/embeddings/%sembeddings_m_%d_kappa_%d_samples_%d_responses_%d_nq_%d_na_%d%s%s%s"

    filename = filename % (salt, m, kappa, samples, responses, number_q,
            number_a, spca, skpca, stsne)

    start = time.time()

    if alternative:
        inds = range(333,999,13)[2:]
        probs = get_alternative_curve_samples(number_q=number_q, number_a=number_a, samples=samples, m=m, inds=inds,
                random=False, rep=rep, sin_angle=kappa)
    else:
        probs = get_curve_samples(number_q=number_q, number_a=number_a, samples=samples, m=m, inds=None,
                random=False, rep=rep, sin_angle=kappa)

    KLs = multi_partite_distance(probs)
    # from pprint import pprint
    # pprint(list(enumerate(KLs)))
    print(max(KLs), min(KLs))

    if True:
        df = get_questionnaires(probs, count_answers=responses, number_q=number_q, number_a=number_a)

        # print("Bartlett Sphericity test: p =", bartlett_sphericity_test(df,
            # df.columns.drop("name_1")))

        # Get Fisher map
        mds, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

        # Compute PCA of the questionnaire
        df.name_1 = df.name_1.astype(int)
        pca, subaks = get_pca(df, df.columns.drop("name_1"), dim=dim,
                align=False)
        if show_tsne:
            tsne = get_t_sne(df, df.columns.drop("name_1"), dim=dim)
        if show_kpca:
            kpca = get_kernel_pca(df, df.columns.drop("name_1"), dim=dim,
                    kernel="sigmoid")
        mca, subaks = get_mca(df, df.columns.drop("name_1"), dim=dim)
        # mfa = get_mfa(df, df.columns.drop("name_1"))

    true_mds = get_true_mds(probs)

    if True:
        # Align the coordinates
        if True:
            pca = align_pca_mds(true_mds, pca)
            kpca = align_pca_mds(true_mds, kpca)
            mca = align_pca_mds(true_mds, mca)
            mds = align_pca_mds(true_mds, mds)
            tsne = align_pca_mds(true_mds, tsne)
            mds_joint = align_pca_mds(true_mds, mds_joint)

    print("Total runtime: %.2f" % (time.time() - start))

    if True:
        print("Plotting theoretical embedding")
        plt.subplot(221)
        plt.scatter(true_mds[:,0], true_mds[:,1], marker='o', label="True FI", c=KLs, s=100, zorder=3)
        plt.plot(true_mds[:,0], true_mds[:,1], label="True FI", lw=3, zorder=2)
        plt.title("Theoretical Embedding")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(label="Multi-partite information")

    if True:
        print("Plotting TSNE")
        if show_tsne:
            tsne_label = "T-SNE"
        mca_label = "MCA" if not show_kpca else "Kernel PCA"
        plt.subplot(222)
        plt.scatter(mca[:,0], mca[:,1], label=mca_label, c=KLs, s=100,
                zorder=3)
        plt.plot(mca[:,0], mca[:,1],  label=mca_label, lw=3, zorder=2)
        mca_corr = corr_between_coords(true_mds, mca)
        plt.title("%s (Corr: %.3f)" % (mca_label, mca_corr))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(label="Multi-partite information")

        plt.subplot(223)
        plt.scatter(mds_joint[:,0], mds_joint[:,1], marker='p', label="FI Joint", c=KLs, s=100, zorder=3)
        plt.plot(mds_joint[:,0], mds_joint[:,1],  label="FI Joint", lw=3,
                zorder=2)
        fi_corr = corr_between_coords(true_mds, mds_joint)
        plt.title("FI (Corr: %.3f)" % fi_corr)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(label="Multi-partite information")

        if show_pca:
            plt.subplot(224)
            plt.scatter(pca[:,0], pca[:,1], label="PCA", c=KLs, s=100,
                    zorder=3)
            plt.plot(pca[:,0], pca[:,1],  label="PCA", lw=3, zorder=2)
            pca_corr = corr_between_coords(true_mds, pca)
            plt.title("PCA (Corr: %.3f)" % pca_corr)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(label="Multi-partite information")

        if show_tsne:
            plt.subplot(224)
            plt.scatter(tsne[:,0], tsne[:,1], label="T-SNE", c=KLs, s=100,
                    zorder=3)
            plt.plot(tsne[:,0], tsne[:,1],  label="T-SNE", lw=3, zorder=2)
            pca_corr = corr_between_coords(true_mds, tsne)
            plt.title("T-SNE (Corr: %.3f)" % pca_corr)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(label="Multi-partite information")
    # plt.legend()
    if save_file:
        print("saving file...")
        # save the plot as pdf
        plt.savefig(filename + ".pdf", bbox_inches="tight")
        # save the data for reproducibility / plot improvement
        f = open(filename + ".pkl", "wb")
        pickle.dump(dict(pca=pca, mds=mds_joint, true=true_mds, KLs=KLs,
            m=m, kappa=kappa, samples=samples, responses=responses,
            number_q=number_q, number_a=number_a, mca=mca, kpca=kpca,
                         tsne=tsne), f)
        f.close()

        print("saved figure as %s.pkl" % filename)

    print("showing plot")
    plt.show()

def multi_partite_distance(probs, Nq=3, Na=3):
    """ Computes the Kullback-Leibler divergence between the joint PDF as
    represented by probs and the marginal distribution for each of these
    joints. This distance is also called the Multi-partite information.

    Args:
        probs (TODO): TODO

    Kwargs:
        Nq (TODO): TODO

        Na (TODO): TODO

    Returns: TODO

    """
    probs = probs ** 2
    ans_dict = get_ans_dict(number_q=Nq, number_a=Na)
    inv_dict = defaultdict(list)
    for k in ans_dict.keys():
        for i in range(Nq):
            inv_dict[(i, ans_dict[k][i])].append(k)

    KLs = []
    for s in probs:
        KL = 0
        marginals = {}
        for k in inv_dict.keys():
            marginals[k] = s[inv_dict[k]].sum()

        for i in ans_dict.keys():
            marg = 1
            for q, j in enumerate(ans_dict[i]):
                marg *= marginals[(q, j)]
            if marg == 0 and s[i] == 0:
                pass
            else:
                KL += s[i] * np.log2(s[i] / marg)

        KLs.append(KL)

    return array(KLs)

if __name__ == "__main__":
    # plot_fi_pca_vs_m()
    # plot_fi_pca_vs_m()

    plot_pca_fi()

    # Calculate an MDS of the simulation as a function of dimension and
    # compute the correlation between the maximal dimension (K-1) and the
    # current dimension. Hopefully a knick will be found.
    if False:
        mpl.rcParams.update({'font.size': 16,
                             'font.family': 'STIXGeneral',
                             'mathtext.fontset': 'stix'})
        start = time.time()
        K = 20
        responses = 25
        m = 3
        Nq, Na = 3, 3
        kappa = 0

        fname = \
        "../paper/images/scree/scree_kappa_{}_m_{}_samples_{}_resp_{}_Nq_{}_Na_{}".format(\
                kappa, m, K, responses, Nq, Na)

        save_file = True

        samples = get_curve_samples(samples=K, m=m, number_q=Nq, number_a=Na,
                sin_angle=kappa)
        df = get_questionnaires(samples, number_q=Nq, number_a=Na,
                count_answers=responses)

        max_dim = K - 1
        _, _, max_mds, stress = compute_mds(df, compute_joint=True, dim=max_dim,
                return_stress=True)

        true_mat = get_true_dist_mat(samples)

        corr_list = []
        corr_with_true = []
        stress_list = []
        for d in range(1, max_dim):
            _, _, mds, stress = compute_mds(df, compute_joint=True, dim=d,
                    return_stress=True)
            corr_list.append(corr_between_coords(mds, max_mds))
            corr_with_true.append(corr_to_dist(true_mat, mds))
            stress_list.append(stress)

        print("Calculation took {} seconds".format(time.time() - start))
        line1, = plt.plot(range(1, max_dim), corr_list, label="Correlations to max dim")
        line2, = plt.plot(range(1, max_dim), corr_with_true, label="Correlation with true dist mat")
        plt.xlabel("MDS Dimension")
        plt.ylabel("Correlation Coefficient")
        ax2 = plt.gca().twinx()
        line3, = ax2.plot(range(1, max_dim), stress_list, label="Stress", color="red")
        plt.ylabel("Stress")
        lns = [line1, line2, line3]
        lbls = [l.get_label() for l in lns]
        plt.legend(lns, lbls, loc='center right')
        if save_file:
            with open(fname + ".pkl", "wb") as f:
                pickle.dump(dict(
                    m = m,
                    kappa = kappa,
                    Nq = Nq,
                    Na = Na,
                    samples = K,
                    corr_list = corr_list,
                    corr_with_true = corr_with_true,
                    stress_list = stress_list,
                    x = list(range(1, max_dim))
                    ), f)
            plt.savefig(fname + ".pdf", bbox_inches="tight")
            print("Saved file as {}.pdf".format(fname))
        plt.show()

    if False:
        start = time.time()

        Nq, Na = 3, 3
        m = 3
        samples = 20
        responses = 50

        samples = get_curve_samples(samples=samples, m=m, number_q=Nq, number_a=Na)
        df = get_questionnaires(samples, number_q=Nq, number_a=Na,
                count_answers=responses)

        get_mfa(df, df.columns.drop("name_1"))

        end = time.time()
        print("Calculation took %.2f seconds" % (end - start))

    if False:
        samples = get_curve_samples(samples=100, m=m)
        KLs = multi_partite_distance(samples)

        theor_mds = get_true_mds(samples)

        plt.scatter(theor_mds[:,0], theor_mds[:,1], c=KLs, s=100)
        plt.colorbar()
        plt.show()


    # plot_fi_pca_vs_resp()
    # plot_fi_pca_vs_samples()

    # print(get_curved_fisher_distance(0.2, 0.3, 2, 3))

    # get_curved_mds()