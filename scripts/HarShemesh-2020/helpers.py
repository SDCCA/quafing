#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed
import mca
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.spatial.distance import pdist, squareform
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

mpl.rcParams.update({'font.size': 22,
                    'font.family': 'STIXGeneral',
                    'mathtext.fontset': 'stix'})




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

def get_hellinger_dist_mat(data, columns, permute=False, dist_type='mean',
        get_fisher=False):
    """
    Computes the Hellinger distance matrix from the DataFrame `data`, assuming
    the data to come from a categorical distribution. Only the columns that
    appear in `columns` are taken into account. If permute is True, returns a
    distance matrix based on the permutation of the Subaks by permuting the
    labels in the `name_1` column.

    Args:
        data (pd.DataFrame): The questionnaire data, that can be grouped by
                             "name_1"

        columns (list like): The columns to take into account.

    Kwargs:
        permute (Bool, False): Shuffle the names for permutation test purposes?

        dist_type ('mean', 'independent'): What type of Hellinger distance to
            compute? 'mean' returns the mean of all questions, 'independent'
            returns the Hellinger distance obtained by assume all questions
            are independent of each other (so the PDF factorizes).

        get_fisher (Bool, True): Return the Fisher information? This is
            accurate based on the relation FI = 4 * arcsin(dH/2).
            See R. E. Kaas, Statistical Science, 4 (3) 188-234 (1989)

    Returns: A distance matrix between all Subaks.

    """
    assert isinstance(data, pd.DataFrame)
    assert dist_type in ('mean', 'independent')

    df = data.copy()
    
    if permute:
        df.name_1 = np.random.permutation(df.name_1)

    # Create indicator columns for all discrete data and join them
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(df[c], prefix=c))

    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df.name_1).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df.name_1).sum()
    prob = sum_df.divide(response_count, "index")
    
    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)

    # Get all pairs of Subaks (without return)
    all_pairs = combinations(sqrt_prob.index, 2)

    # Compute the Hellinger distance for all pairs
    res = {}
    res_mean = {}
    for c1, c2 in all_pairs:
        s1 = sqrt_prob.ix[c1]
        s2 = sqrt_prob.ix[c2]
        subak_diff = 0.5 * (s1 - s2)**2

        # This gives groups of questions
        questions = subak_diff.groupby(lambda x: "_".join(x.split("_")[:-1])).sum()

        H = 1.0 if np.any(questions==1) else np.sqrt(1 - (1 - questions).prod())
        H_mean = np.sqrt(questions).mean()
        if get_fisher:
            H = 4 * np.arcsin(H)
            H_mean = 4 * np.arcsin(H_mean)
        if c1 in res:
            res[c1][c2] = H
            res_mean[c1][c2] = H_mean
        else:
            res[c1] = {c2: H, c1: 0.0}
            res_mean[c1] = {c2: H_mean, c1: 0.0}
        if c2 in res:
            res[c2][c1] = H
            res_mean[c2][c1] = H_mean
        else:
            res[c2] = {c1: H, c2: 0.0}
            res_mean[c2] = {c1: H_mean, c2: 0.0}
    hellinger_dist_mat = pd.DataFrame(res)
    hellinger_mean_dist_mat = pd.DataFrame(res_mean)
    if dist_type == 'mean':
        return hellinger_mean_dist_mat
    return hellinger_dist_mat

def get_hypersphere_angles(data, column):
    """ Returns a set of angles for a hypersphere defined by the square root
    of the probabilities. This is done for a single question only (at the
    moment).

    Args:
        data (TODO): TODO
        column (TODO): TODO

    Returns: TODO

    """
    assert isinstance(data, pd.DataFrame)
    assert column in data.columns

    df = data.loc[:, ['name_1', column]].copy()
    
    # Create indicator columns for all discrete data and join them
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    ind_df = ind_df.join(pd.get_dummies(df[column], prefix=column))

    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df.name_1).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df.name_1).sum()
    prob = sum_df.divide(response_count, "index")
    
    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)

    # Compute the number of necessary angles
    angle_count = sqrt_prob.columns.shape[0] - 1

def compute_mds(df, dim=2, compute_joint=False, columns=None, get_names=False,
        return_stress=False, return_dist_mat=False):
    """ Computes the MDS embedding of the questionnaire results based on
    Fisher information distance.

    Args:
        df (DataFrame): The results of the questionnaire from Steve

    Kwargs:
        dim (int, 2): The dimensionality of the MDS estimation

        compute_joint (Bool, False): Should we use the joint PDF (assuming
            independence) to compute the distance?

        columns (list-like, None): If provided, which columns to take into
            account in the calculation?

        get_names (Bool, False): If True, the function will also return a list
            of names as provided in the name_1 column.

    Returns: Either a list of coordinates with the MDS, or, if compute_joint
        is True, two lists, one with mean MDS and one with Joint MDS.

    """
    # Estimate probabilities from questionnaire
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    if columns == None:
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

    # Get all pairs of Subaks (without return)
    all_pairs = combinations(sqrt_prob.index, 2)

    res = {}
    res_joint = {}
    for c1, c2 in all_pairs:
        s1 = sqrt_prob.ix[c1]
        s2 = sqrt_prob.ix[c2]

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

        # A faster way to get the joint
        FI_joint = (np.arccos(questions.prod())) ** 2

        # c1i = int(c1)
        # c2i = int(c2)

        if c1 in res:
            res[c1][c2] = FI
            res_joint[c1][c2] = FI_joint
        else:
            res[c1] = {c2: FI, c1: 0.0}
            res_joint[c1] = {c2: FI_joint, c1: 0.0}
        if c2 in res:
            res[c2][c1] = FI
            res_joint[c2][c1] = FI_joint
        else:
            res[c2] = {c1: FI, c2: 0.0}
            res_joint[c2] = {c1: FI_joint, c2: 0.0}

    dist_mat = pd.DataFrame(res).sort_index().sort_index(axis=1)
    dist_mat_joint = pd.DataFrame(res_joint).sort_index().sort_index(axis=1)

    # Perform the MDS
    mds_coords = MDS(n_components=dim, dissimilarity='precomputed').fit_transform(dist_mat)
    if return_stress:
        mds = MDS(n_components=dim, dissimilarity='precomputed', max_iter=600,
                n_init=30).fit(dist_mat_joint)
        mds_coords_joint = mds.embedding_
        stress = mds.stress_
    else:
        mds_coords_joint = MDS(n_components=dim, dissimilarity='precomputed',
                max_iter=600, n_init=30).fit_transform(dist_mat_joint)
    
    if (not compute_joint) and (not get_names):
        return mds_coords

    result = [mds_coords]

    if compute_joint:
        result.append(mds_coords_joint)

    if get_names:
        result.append(dist_mat.index)

    if return_stress:
        result.append(stress)

    if return_dist_mat:
        result.append(dist_mat_joint if compute_joint else dist_mat)

    return result

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

def compute_correlation_with_mds(coords, dist_mat):
    """ This computes the Pearson product-moment correlation between the
    distances obtained by MDS and the original distance matrix used to compute
    these coordinates. This can indicate to what extent the dimensionality of
    the data is sufficient.

    Args:
        coords (TODO): TODO
        dist_mat (TODO): TODO

    Returns: TODO

    """
    coords_dist = pdist(coords)
    dist_mat_condensed = squareform(dist_mat)
    corr = np.corrcoef(coords_dist, dist_mat_condensed)
    return corr[0,1]

def get_mds(df, columns, method='FI', dim=2, get_fisher=False,
        return_corr=False):
    """ Computes the MDS of the distance matrix obtained for the given df and
    columns.

    Args:
        df (DataFrame): The data
        columns (list): list of columns to take into account

    Kwargs:
        method (str, 'FI'): Should the mean or independent Hellinger distance
            be used?

        dim (int, 2): What should be the dimensionality of the embedding?

        return_corr (Bool, False): If set to True, the correlation between the
            coordinate distances and the hellinger distance matrix will be
            return together with the other things.


    Returns: coordinates of the embedding and names of Subaks

    """
    if method == 'FI+PCA':
        n_comp = 19
        print("Method is FI+PCA!")
    else:
        n_comp = dim

    mds = MDS(n_components=n_comp, random_state=np.random.seed(0), dissimilarity='precomputed')
    if method in ['FI', 'FI+PCA']:
        dist_mat = get_hellinger_dist_mat(df, columns, get_fisher=get_fisher)
    else:
        dist_mat = get_hellinger_dist_mat(df, columns, dist_type='independent', get_fisher=get_fisher)
    coords = mds.fit_transform(dist_mat)
    if method == "FI+PCA":
        coords = PCA(n_components=2).fit_transform(coords)
    if return_corr:
        corr = compute_correlation_with_mds(align_coords(coords), dist_mat)
        return align_coords(coords), dist_mat.index, corr
    return align_coords(coords), dist_mat.index

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

def get_mca(df, columns, permute=False):
    """ Computes the Multiple Correspondence Analysis for the given data.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        dim (int): The number of dimensions to return

    Returns: TODO

    """
    assert isinstance(df, pd.DataFrame)
    assert len(columns) > 1
    
    data = df.copy()

    if permute:
        data.name_1 = np.random.permutation(data.name_1)

    # To avoid a bug with MCA using Categorical index
    # https://github.com/esafak/mca/issues/11
    if "harvest_discrete_9" in columns:
        data["harvest_discrete_9"] = pd.to_numeric(data["harvest_discrete_9"],
                downcast="integer")

    MCA = mca(data, cols=list(columns)).fs_r(N=2)
    subaks = sorted(df.name_1.unique())
    mca_coords = np.zeros(shape=(len(subaks), 2))
    for i, s in enumerate(subaks):
        # Compute the position of the Subak in the MCA space
        ind = df[df.name_1 == s].index
        mca_coords[i,:] = np.mean(MCA[ind], axis=0)

    mca_coords = align_coords(mca_coords)
    return mca_coords, subaks

def normalize_coordinates(change, reference):
    """ Normalizes the coordinates of change such that the largest distance in
    change will equal the largest distance in reference.

    Args:
        change (20x2 ndarray): Array of coordinates to scale
        reference (20x2 ndarray): Reference array

    Returns: (20x2 ndarray) The scaled coordinates

    """
    max_dist_change = pairwise_distances(change).max()
    max_dist_reference = pairwise_distances(reference).max()
    new_change = change * max_dist_reference / max_dist_change
    return new_change

def get_embedding(df, columns, method='FI', dim=2, get_fisher=False):
    """ Returns an embedding based on either PCA or Hellinger/MDS.
    
    """
    assert method in ['FI', 'FI+PCA', 'FI ind', 'PCA', 'MCA']

    if 'FI' in method:
        return get_mds(df, columns, method, dim=dim, get_fisher=get_fisher)
    elif method == 'MCA':
        return get_mca(df, columns)
    else:
        return get_pca(df, columns)

def get_subak_dist(df, subak, column):
    """ Returns the distribution of responses for a specific Subak for a
    specific question

    Args:
        df (TODO): TODO
        subak (TODO): TODO
        column (TODO): TODO

    Returns: TODO

    """
    assert isinstance(df, pd.DataFrame)
    assert "name_1" in df.columns
    assert column in df.columns
    assert subak in df.name_1.unique()

    ind_df = pd.get_dummies(df[df.name_1 == subak][column])
    prob = ind_df.sum() / len(ind_df)
    complete_ind = df[column].unique()

    return prob.reindex(complete_ind, fill_value=0).sort_index()

def get_permutations(df, columns, method='FI', count=100):
    """ Performs `count` permutations and returns a list of all resulting
    coordinates (without identifying the various labels from the permutations)

    Args:
        df (pd.DataFrame): Original data DataFrame
        columns (list): List of columns to include
    Kwargs:
        count (int, 100): Number of repetitions

    Returns: List of coordinates ndarray, one for each permutation.

    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(count, int)

    results = []
    for i in range(count):
        if method in ['FI', 'FI+PCA', 'FI ind']:
            dist_mat = get_hellinger_dist_mat(df, columns, permute=True,
                    get_fisher=True)
            if method == 'FI+PCA':
                mds = MDS(n_components=19, random_state=np.random.seed(i), dissimilarity='precomputed')
            else:
                mds = MDS(n_components=2, random_state=np.random.seed(i), dissimilarity='precomputed')
            coords = mds.fit_transform(dist_mat)
            if method == 'FI+PCA':
                coords = PCA(n_components=2).fit_transform(coords)
            results.append(coords)
        elif method == "MCA":
            coords, subaks = get_mca(df, columns, permute=True)
            results.append(coords)
        elif method == "PCA":
            coords, subaks = get_pca(df, columns, permute=True)
            results.append(coords)

    return results

def get_clustering(df, columns, k=4):
    """ Computes a KMeans clustering on the 19 dimensional space of the Subaks
    and returns the k clusters it found

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        k (TODO): TODO

    Returns: TODO

    """

    # Get the distance matrix and 19 dimensional embedding
    dist_mat = get_hellinger_dist_mat(df, columns, dist_type='mean')
    dim = len(dist_mat.index) - 1
    mds = MDS(n_components=dim, random_state=np.random.seed(0), dissimilarity='precomputed')
    coords = mds.fit_transform(dist_mat)

    # Perform the clustering
    kmeans = KMeans(n_clusters=k).fit_predict(coords)
    clusters = pd.DataFrame(kmeans, index=dist_mat.index, columns=['Clusters'])
    return clusters

def get_center_distribution(df, columns, cluster):
    """ Computes the distribution of answers for the questions specified in
    `columns` for the collection of cluster subaks.

    Args:
        df (TODO): TODO
        columns (TODO): TODO
        cluster (TODO): TODO

    Returns: TODO

    """
    # Get the data for all Subaks in the cluster
    data = df[df.name_1.isin(cluster)].copy()
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(data.name_1, how="right")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(data[c], prefix=c))

    ind_df = ind_df.drop('name_1', axis=1)

    # Turn this into a probabilities matrix for each possible response
    response_count = ind_df.shape[0]
    sum_df = ind_df.sum()
    prob = sum_df.divide(response_count, "index")
    return prob

def get_distances_from_center(df, columns, center_prob, method='mean'):
    """ Computes a list of distances from the center cluster

    Args:
        df (TODO): TODO
        columns (TODO): TODO
        center_dist (TODO): TODO

    Kwargs:
        method (TODO): TODO

    Returns: TODO

    """
    subaks = sorted(df.name_1.unique())
    # Create indicator columns for all discrete data and join them
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(df.name_1, how="right")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(df[c], prefix=c))

    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df.name_1).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df.name_1).sum()
    prob = sum_df.divide(response_count, "index")
    
    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)
    sqrt_center_prob = np.sqrt(center_prob)

    # Compute the Hellinger distance for all pairs
    res = {}
    res_mean = {}
    for c in subaks:
        s = sqrt_prob.ix[c]
        subak_diff = 0.5 * (s - sqrt_center_prob)**2

        # This gives groups of questions
        questions = subak_diff.groupby(lambda x: "_".join(x.split("_")[:-1])).sum()

        H = 1.0 if np.any(questions==1) else np.sqrt(1 - (1 - questions).prod())
        H_mean = questions.mean()
        res[c] = H
        res_mean[c] = H_mean

    if method == "mean":
        return res_mean
    return res

def get_distribution_k_means(df, columns, k=4, max_rep=1000, n_init=30):
    """ Computes the k-means clustering algorithm based on Hellinger distance
    and a mean Subak which is the mean of all Subaks, computed by aggregating
    the answers to all Subaks in the cluster and then computing the distance
    from it to the other Subaks.

    Args:
        df (DataFrame): Questionnaire data
        columns (pd.Index): List of relevant columns for the calculation

    Kwargs:
        k (int): number of clusters

    Returns: A list of cluster labels for the Subaks

    """
    assert isinstance(k, int) and k > 0
    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, (pd.Index, list))
    best_inertia = np.inf

    for j in range(n_init):
        # Get random assignment of cluster centers (i.e. random Subaks)
        subaks = sorted(df.name_1.unique())
        centers = np.random.choice(subaks, size=k, replace=False)
        clusters = {}
        for i, c in enumerate(centers):
            clusters[i] = [c]

        rep = 0
        while rep < max_rep:
            # Compute distances from each center
            distributions = {}
            distances = {}
            for i in clusters:
                distributions[i] = get_center_distribution(df, columns, clusters[i])
                distances[i] = get_distances_from_center(df, columns, distributions[i])

            # Construct clusters based on minimal distance
            new_clusters = {i: [] for i in range(k)}
            for s in subaks:
                # find the cluster with minimal distance
                distance = np.inf
                candidate = None
                for i in clusters:
                    if distances[i][s] < distance:
                        distance = distances[i][s]
                        candidate = i

                new_clusters[candidate].append(s)

            rep += 1
            if new_clusters == clusters:
                break
            clusters = new_clusters

        # Compute the inertia of the result (sum of distances from centers)
        inertia = 0
        for i in new_clusters:
            for s in new_clusters[i]:
                inertia += distances[i][s]

        if inertia < best_inertia:
            best_clusters = new_clusters
            best_inertia = inertia

    res = []
    ind = []
    for i in best_clusters:
        for s in best_clusters[i]:
            res.append(i)
            ind.append(s)
    return pd.DataFrame(res, index=ind, columns=['Cluster'])

def get_pairwise_hellinger(df, ind_df, groupby='name_1', kind='mean',
        permute=False):
    """ Computes the Hellinger distance between a pair of Subaks for the given
    columns.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        groupby (TODO): TODO

    Returns: TODO

    """
    assert kind in ["mean", "independent"]

    if permute:
        df.loc[:, groupby] = np.random.permutation(df[groupby])
        ind_df.loc[:, groupby] = df[groupby]
    # Turn this into a probabilities matrix for each possible response
    response_count = df.groupby(df[groupby]).apply(lambda x: len(x))
    sum_df = ind_df.groupby(ind_df[groupby]).sum()
    prob = sum_df.divide(response_count, "index")
    
    # Compute the square root of the probabilities
    sqrt_prob = np.sqrt(prob)
    c1 = df[groupby].unique()[0]
    c2 = df[groupby].unique()[1]

    s1 = sqrt_prob.ix[c1]
    s2 = sqrt_prob.ix[c2]
    subak_diff = 0.5 * (s1 - s2)**2

    # This gives groups of questions
    questions = subak_diff.groupby(lambda x: "_".join(x.split("_")[:-1])).sum()

    H = 1.0 if np.any(questions==1) else np.sqrt(1 - (1 - questions).prod())
    H_mean = questions.mean()

    if kind == "mean":
        return H_mean
    else:
        return H

def parallel_function(params):
    return get_pairwise_hellinger(params[0], params[1], permute=True)

def get_p_value(df, columns, s1, s2, groupby='name_1', repetitions=100000):
    """ Computes a p-value for the Hellinger distance between Subaks s1 and s2
    by using a permutation test repeated `repetitions` times.

    Args:
        df (DataFrame): Questionnaire data
        columns (list): List of columns of df to participate in the
                        computation.
        s1 (str): Subak 1
        s2 (str): Subak 2

    Kwargs:
        repetitions (int): Number of times to repeat the permutation test

    Returns: The p-value from the computation

    """
    data = df[(df[groupby] == s1) | (df[groupby] == s2)]
    ind_df = pd.DataFrame()
    ind_df = ind_df.join(data[groupby], how="right")
    for c in columns:
        ind_df = ind_df.join(pd.get_dummies(data[c], prefix=c))

    H = get_pairwise_hellinger(data, ind_df)
    H_perm = []
    params = []
    for i in range(repetitions):
        params.append([data, ind_df])

    pool = multiprocessing.Pool(processes=7)
    H_perm = pool.map(parallel_function, params)

    # Compute the p-value
    H_perm = np.array(H_perm)
    p = np.sum(H_perm >= H) / repetitions

    return H, H_perm, p

def get_distributional_clustering(df, columns, cutoff=0.05, repetitions=1000):
    """  Makes a hypothesis test for each pair of Subaks to test whether they
    can be distinguished by using permutation tests. The probability of the
    obtained Hellinger distance is computed by repeating the permutation tests
    many times and computing the various resulting Hellinger distances. 
    Then, using the cutoff, we assign all Subaks that cannot be distinguished
    according to the cutoff to the same cluster. This results in a clustering
    where the number of clusters is not imposed and is, in general, more
    consistent with the statistical nature of the underlying work.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        cutoff (TODO): TODO
        repetitions (TODO): TODO

    Returns: TODO

    """
    pass

def load_data(fname="data/subaks_preprocessed.xlsx"):
    df = pd.read_excel(fname, header=0)

    # Question excluded by Hendrik/Steve
    df.drop(['owned_3'], axis=1, inplace=True) 

    # Turn the harvest variable to discrete by taking the quartiles
    df["harvest_discrete_9"] = pd.qcut(
        df["harvest_kg/ha_9"], 4, labels=list("1234"))


    def f(x):
        if x < .25: return 0
        if x < .5: return 1
        if x < .75: return 2
        return 3
    df["sharecrop_discrete_4"] = df.sharecrop_4.apply(f)

    # At first take only discrete data columns
    columns = df.columns.drop(
        ["name_1", "number_2", "sharecrop_4", "harvest_kg/ha_9"])

    # Sort columns based on question number
    columns = pd.Index(sorted(columns, key = lambda x: int(x.split("_")[-1])))
    
    return df, columns

def load_metadata(fname="./data/subak_metadata.csv"):
    smd = pd.read_csv(fname, index_col=0)
    return smd

def get_probs(alpha, beta, f=0.3):
    """ Computes the probabilities necessary to simulate the responses to
    "follow meetings".

    Args:
        alpha (int or array): The ratio of p2 to p1
        beta (int or array): The argument of the Heaviside function

    Kwargs:
        f (float): The intensity of the rise of p3

    Returns: a tuple (p1, p2, p3) of either scalars or a numpy array
        (depending on input)
        
    """
    if len(alpha) > 1 and len(beta) > 1:
        assert len(alpha) == len(beta)

    # Heaviside function of beta
    Hb = 0.5 * (np.sign(beta) + 1)

    p3 = f * Hb * alpha
    p1 = (1 - p3) / (1 + alpha)
    p2 = alpha * p1
    return (p1, p2, p3)

def get_simulated_data(N_subaks=20, N_answers=25, N_q=5, a_low=0, a_high=1,
        prob_b=0.2, f=0.3):

    subaks = np.concatenate([[str(i)] * N_answers for i in range(N_subaks)])
    data = {"name_1": subaks}

    for i in range(N_q):
        alphas = np.random.uniform(a_low, a_high, size=N_subaks)
        betas = np.random.uniform(prob_b-1, prob_b, size=N_subaks)
        p1, p2, p3 = get_probs(alphas, betas, f=f)
        probs = zip(p1, p2, p3)

        # Create the answers according to the distributions
        answers = []
        for p in probs:
            p1 = p[0]
            p2 = p[1]
            p3 = p[2]
            samples = np.random.uniform(size=N_answers)
            ans = 1 * (samples > p1) + 2 * (samples - p1 > p2) + 3 * (samples - p1 - p2 > p3)
            answers.extend(ans)

        data["question_%d" % i] = answers

    df = pd.DataFrame(data)
    return df

def get_fisher_plots(df, columns, dim=3):
    """ Computes an MDS embedding and returns a plot with enough subplots to
    show all dimensions requested. For example, if dim=2 one plot is returned.
    For dim=3, three plots will be returned, for all three planes.
    For higher dimensions similarly plots will be returned.

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        dim (TODO): TODO

    Returns: TODO

    """
    coords, subaks = get_mds(df, columns, dim=dim, get_fisher=True)
    fig, axes = plt.subplots()

def update_position(e, fig, ax, subaks, mds_joint, labels):
    for i in range(len(subaks)):
        x2, y2, _ = proj3d.proj_transform(mds_joint[i, 0], mds_joint[i, 
            1], mds_joint[i, 2], ax.get_proj())
        labels[i].xy = x2,y2
        labels[i].update_positions(fig.canvas.renderer)
    fig.canvas.draw()

def plot_dimension_variance(df, columns, dim=8, use_pca=False):
    """
    Gets a set of coordinates in d dimensions and returns the variance of the
    coordinates in each of the dimensions, to get an estimate where the
    interesting things are happening.

    """
    coords, subaks = get_mds(df, columns, dim=dim, get_fisher=True)

    if use_pca:
        # transform the coordinates using PCA to have their variances sorted
        # by importance
        pca_coords = PCA(n_components=dim).fit_transform(coords)

        variances = pca_coords.var(axis=0)
        label = "Using PCA"
    else:
        variances = coords.var(axis=0)
        label = "Not using PCA"
    plt.plot(variances, label=label)
    plt.scatter(range(dim), variances)
    plt.legend()
    plt.show()
    return pca_coords if use_pca else coords

def plot_interactive_mds_pca(df, columns, dim=19, color_elevation=False,
        savefig=False, fname="/home/omri/repos/pnas/images/subaks/with_pca.pdf"):
    """
    Plots the subaks on an interactive plot (that shows their names on hover).
    Also compares with other methods (perhaps)
    """
    if not savefig:
        from annotater import AnnoteFinder
    import matplotlib as mpl
    from adjustText import adjust_text
    coords, subaks = get_mds(df, columns, dim=dim, get_fisher=True)
    pca_coords = PCA(n_components=2).fit_transform(coords)
    pca_emb, subaks2 = get_mca(df, columns)
    _, pca_emb, z = procrustes(pca_coords, pca_emb)
    #def on_pick(event):
        #ind = event.ind
        #print(subaks[ind])

    #plt.ion()

    meta = load_metadata()
    elv = meta[meta.columns[3]]
    subak_elv = ["%s (%d m)" % (s, elv[s]) for s in subaks]
    if color_elevation:
        colors = elv[subaks]
        normed = mpl.colors.Normalize(min(colors), max(colors))
        mapper = mpl.cm.ScalarMappable(normed, "Paired")
        colors = mapper.to_rgba(colors)
    else:
        #colors = ['blue' if s in ['']]
        colors = range(len(subaks))
        colors = ['blue' if s in ['Aban, Darmasaba', 'Subak Dukuh, Kapal', 'Subak Tegan, Kapal',  'Teba, Desa Tangeb'] else 'green' 
                if s in ['Betuas, Keramas', 'Mantring', 'Selukat', 'Kulub Atas'] else 'red' for s in subaks]


    fig = plt.figure()
    #gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[5, 5, 1])
    ax1 = plt.subplot2grid((6,2), (0,0), rowspan=5)
    ax2 = plt.subplot2grid((6,2), (0,1), rowspan=5)
    ax3 = plt.subplot2grid((6,2), (5,0), colspan=2)
    #ax1 = plt.subplot(gs[0])
    #ax2 = plt.subplot(gs[1])
    #ax3 = plt.subplot(gs[2])
    sc = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], cmap="Paired",
            c=colors, edgecolor='k', s=81)
    sc2 = ax2.scatter(pca_emb[:, 0], pca_emb[:, 1], cmap="Paired",
            c=colors, edgecolor='k', s=81)
    #fig.canvas.mpl_connect('pick_event', on_pick)
    if not savefig:
        af = AnnoteFinder(pca_coords[:, 0], pca_coords[:, 1], subak_elv, ax=ax1)
        fig.canvas.mpl_connect('button_press_event', af)
        af2 = AnnoteFinder(pca_emb[:, 0], pca_emb[:, 1], subak_elv, ax=ax2)
        fig.canvas.mpl_connect('button_press_event', af2)
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_aspect('equal', adjustable='datalim')
    ax1.set_title("FI", fontsize=30)
    ax2.set_title("MCA", fontsize=30)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    if color_elevation:
        mapper.set_array(elv[subaks])
        plt.colorbar(mapper, cax=ax3, orientation='horizontal',
                label="Elevation [m]")
        #ax3.set_title("Elevation [m]")

    # Add labels for all Subaks with adjust_text
    texts = []
    for (x, y, s) in zip(pca_coords[:, 0], pca_coords[:, 1], subaks):
        if s in ['Betuas, Keramas', 'Selukat', 'Pakudui', 'Mantring', 
                'Kulub Atas', 'Subak Dukuh, Kapal']:
            texts.append(ax1.text(x, y, s, size=24))
    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='->', color='r'))

    texts = []
    for (x, y, s) in zip(pca_emb[:, 0], pca_emb[:, 1], subaks):
        if s in ['Betuas, Keramas', 'Selukat', 'Pakudui', 'Mantring', 
                'Kulub Atas', 'Subak Dukuh, Kapal']:
            texts.append(ax2.text(x, y, s, size=24))
    adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle='->', color='r'))

    plt.tight_layout()
    if savefig:
        plt.savefig(fname, bbox_inches='tight')
        print("Saved file to {}".format(fname))
    else:
        plt.show()

def make_subak_scree_plot(df, columns, with_pca=False, verbose=True):
    """ Makes a scree plot for the Subaks with FI+PCA method

    Args:
        df (TODO): TODO
        columns (TODO): TODO

    Kwargs:
        with_pca (TODO): TODO

    Returns: TODO

    """
    save_file = False
    fname = "/home/omri/Dropbox/PhD/pnas/images/subaks/scree_subaks_new.pdf"
    max_dim = len(df.name_1.unique()) - 1

    corr_with_max_dim = []
    corr_pca = []
    stress_all = []
    corr_mds = []
    corr_pca_dist_mat = []
    _, mds_max, stress_max, dist_mat = compute_mds(df, compute_joint=True,
            columns=columns,
            dim=max_dim-1, return_stress=True, return_dist_mat=True)
    pca = PCA(n_components=2)
    if with_pca:
        mds_pca_max = pca.fit_transform(mds_max)
    for d in range(1, max_dim-1):
        _, mds, stress = compute_mds(df, compute_joint=True,
                columns=columns,
                dim=d, return_stress=True)
        if with_pca and d > 2:
            mds_pca = pca.fit_transform(mds)
            corr_pca.append(corr_between_coords(mds_pca_max, mds_pca))
            corr_pca_dist_mat.append(compute_correlation_with_mds(mds_pca,
                dist_mat))
        corr_with_max_dim.append(corr_between_coords(mds_max, mds))
        corr_mds.append(compute_correlation_with_mds(mds, dist_mat))
        # corr_between_coords
        stress_all.append(stress)
        if verbose:
            sys.stdout.write("\rFinished {} from {}".format(d, max_dim - 2))

    print('\n')
    stress_all.append(stress_max)
    corr_with_max_dim.append(1.0)
    corr_pca.append(1.0)
    corr_mds.append(compute_correlation_with_mds(mds_max, dist_mat))
    corr_pca_dist_mat.append(compute_correlation_with_mds(mds_pca_max, dist_mat))
    plt.plot(range(1, max_dim), stress_all, label="Stress")
    ax = plt.gca()
    plt.xlabel("Dimension")
    plt.xticks([2,4,6,8,10,12,14,16,18])
    plt.ylabel("Stress")
    ax2 = ax.twinx()
    ax2.plot(range(1, max_dim),  corr_with_max_dim, label="Correlation with Max")
    if with_pca:
        ax2.plot(range(3, max_dim),  corr_pca, label="Correlation with PCA")
        ax2.plot(range(3, max_dim),  corr_pca_dist_mat, label="PCA with dist mat")
    ax2.plot(range(1, max_dim), corr_mds, label="Corr with dist_mat")
    ax2.set_ylabel("Pearson Correlation")
    plt.legend(loc="best")
    if save_file:
        plt.savefig(fname, bbox_inches="tight")
        print("Saved scree plot as {}".format(fname))
    else:
        plt.show()

if __name__ == "__main__":

    # print(get_simulated_data())

    df, columns = load_data()
    make_subak_scree_plot(df, columns, with_pca=True)

    sys.exit(0)

    #plot_dimension_variance(df, columns, use_pca=True)
    # plot_interactive_mds_pca(df, columns, dim=8, color_elevation=False,
            # savefig=False)
    #embed()
    # sys.exit(0)

    reduced_cooperative = ['inherited_5', 'synchronize_16',
            'attend_meeting_17', 'work_participation_18',
            'follow_decision_20', 'selection_of_subak_head_27', 'fines_28',
            'collective_work_30', 'reading_rules_31', 'condition_of_subak_40',
            'follow_meeting_41']

    reduced_resource = ['sharecrop_discrete_4', 'water_shortages_21',
            'self_shortages_22', 'class_45']

    reduced_disharmony = [
       'theft_frequency_25', 'social_conflicts_26',
       'social_problems_43', 'caste_44']

    cols = reduced_cooperative
    cols.extend(reduced_resource)
    cols.extend(reduced_disharmony)

    # Perform outlier detection
    if False:
        cols_list = [('All', cols), ('Cooperative', reduced_cooperative),
                ('Resources', reduced_resource), ('Disharmony',
                    reduced_disharmony)]
        for l, c in cols_list:
            _, mds, subaks = compute_mds(df, compute_joint=True, columns=c,
                    dim=6, get_names=True)
            frst = IsolationForest(contamination=0.25, n_estimators=200).fit(mds)
            print("Outliers with {}: {}".format(l, "; ".join(subaks[frst.predict(mds) < 0])))
            
    ddd = False
    save_file = True
    if False:  # Plot MDS of reduced descriptors and of all combined
        fig = plt.figure(figsize=(8, 8 * np.sqrt(2)))
        plt.subplot(421)
        if ddd:
            ax1 = fig.add_subplot(421, projection='3d')
            _, mds1, subaks = compute_mds(df, compute_joint=True, columns=cols, dim=3,
                    get_names=True)
            ax1.scatter(mds1[:, 0], mds1[:, 1], mds1[:, 2], c=range(len(subaks)))
        else:
            ax1 = plt.gca()
            _, mds1, subaks = compute_mds(df, compute_joint=True, columns=cols, dim=2,
                    get_names=True)
            mds1 = align_coords(mds1)
            plt.scatter(mds1[:,0], mds1[:,1], c=range(mds1.shape[0]), s=80,
                    edgecolor="k")
        plt.title("All descriptors (FI)")
        # ax1.set_aspect("equal")

        plt.subplot(422)
        mca1, subaks = get_mca(df, columns=cols)
        # Align MCA1 with MDS1 as best as possible
        if not ddd:
            d, mca1, tform = procrustes(mds1, mca1)
            mca1 = align_coords(mca1)
        plt.scatter(mca1[:,0], mca1[:,1], c=range(mca1.shape[0]), s=80,
                edgecolor="k")
        plt.title("All descriptors (MCA)")
        ax1a = plt.gca()
        # ax1a.set_aspect(1)
        # ax1a.set_aspect("equal")

        plt.subplot(423)
        _, mds2 = compute_mds(df, compute_joint=True,
                columns=reduced_cooperative, dim=2)
        # Align MDS2 with MDS1 as best as possible
        if not ddd:
            d, mds2, tform = procrustes(mds1, mds2)
            mds2 = align_coords(mds2)
        plt.scatter(mds2[:,0], mds2[:,1], c=range(mds2.shape[0]), s=80,
                edgecolor="k")
        plt.title("Cooperative descriptors (FI)")
        ax2 = plt.gca()
        # ax2.set_aspect("equal")
        
        plt.subplot(424)
        mca2, subaks = get_mca(df, columns=reduced_cooperative)
        # Align MCA2 with MDS1 as best as possible
        if not ddd:
            d, mca2, tform = procrustes(mds1, mca2)
            mca2 = align_coords(mca2)
        plt.scatter(mca2[:,0], mca2[:,1], c=range(mca2.shape[0]), s=80,
                edgecolor="k")
        plt.title("Cooperative descriptors (MCA)")
        ax2a = plt.gca()
        # ax2a.set_aspect("equal")

        plt.subplot(425)
        _, mds3 = compute_mds(df, compute_joint=True,
                columns=reduced_resource, dim=2)
        # Align MDS3 with MDS1 as best as possible
        if not ddd:
            d, mds3, tform = procrustes(mds1, mds3)
            mds3 = align_coords(mds3)
        plt.scatter(mds3[:,0], mds3[:,1], c=range(mds3.shape[0]), s=80,
                edgecolor="k")
        plt.title("Resources descriptors (FI)")
        ax3 = plt.gca()
        # ax3.set_aspect("equal")

        plt.subplot(426)
        mca3, subaks = get_mca(df, columns=reduced_resource)
        # Align MCA3 with MDS1 as best as possible
        if not ddd:
            d, mca3, tform = procrustes(mds1, mca3)
            mca3 = align_coords(mca3)
        plt.scatter(mca3[:,0], mca3[:,1], c=range(mca3.shape[0]), s=80,
                edgecolor="k")
        plt.title("Resources descriptors (MCA)")
        ax3a = plt.gca()
        # ax3a.set_aspect("equal")

        plt.subplot(427)
        _, mds4 = compute_mds(df, compute_joint=True,
                columns=reduced_disharmony, dim=2)
        # Align MDS4 with MDS1 as best as possible
        if not ddd:
            d, mds4, tform = procrustes(mds1, mds4)
            mds4 = align_coords(mds4)
        plt.scatter(mds4[:,0], mds4[:,1], c=range(mds4.shape[0]), s=80,
                edgecolor="k")
        plt.title("Disharmony descriptors (FI)")
        ax4 = plt.gca()
        # ax4.set_aspect("equal")

        plt.subplot(428)
        mca4, subaks = get_mca(df, columns=reduced_disharmony)
        # Align MCA4 with MDS1 as best as possible
        if not ddd:
            d, mca4, tform = procrustes(mds1, mca4)
            mca4 = align_coords(mca4)
        plt.scatter(mca4[:,0], mca4[:,1], c=range(mca4.shape[0]), s=80,
                edgecolor="k")
        plt.title("Disharmony descriptors (MCA)")
        ax4a = plt.gca()
        # ax4a.set_aspect("equal")

        if ddd:
            labels1 = []
        for i in range(len(subaks)):
            if not subaks[i] in ['Betuas, Keramas', 'Selukat', 'Pakudui',
                    'Mantring', 'Kulub Atas', 'Tampuagan Hulu']:
                continue
            if ddd:
                x, y, _ = proj3d.proj_transform(mds1[i, 0], mds1[i, 1], 
                        mds1[i, 2], ax1.get_proj())
                labels1.append(ax1.annotate(subaks[i], xy=(x, y),
                    textcoords='offset points'))
            else:
                ax1.annotate(subaks[i], xy=(mds1[i, 0], mds1[i, 1]))
            ax1a.annotate(subaks[i], xy=(mca1[i, 0], mca1[i, 1]))
            ax2.annotate(subaks[i], xy=(mds2[i, 0], mds2[i, 1]))
            ax2a.annotate(subaks[i], xy=(mca2[i, 0], mca2[i, 1]))
            ax3.annotate(subaks[i], xy=(mds3[i, 0], mds3[i, 1]))
            ax3a.annotate(subaks[i], xy=(mca3[i, 0], mca3[i, 1]))
            ax4.annotate(subaks[i], xy=(mds4[i, 0], mds4[i, 1]))
            ax4a.annotate(subaks[i], xy=(mca4[i, 0], mca4[i, 1]))

        if ddd:
            fig.canvas.mpl_connect('motion_notify_event', lambda e: update_position(e, 
                fig, ax1, subaks, mds1, labels1))

        plt.tight_layout()
        if save_file:
            fname = "../paper/images/subaks/reduced_indicators_less_labels.pdf"
            plt.savefig(fname, bbox_inches="tight")
            print("Saved file as {}".format(fname))
        else:
            plt.show()


    if False: # Makes a scree plot
        save_file = False
        fname = "../paper/images/subaks/scree_subaks.pdf"
        max_dim = len(df.name_1.unique()) - 1

        # _, mds_max = compute_mds(df, compute_joint=True, 
                # columns=cols, dim=max_dim)

        corr_with_max_dim = []
        stress_all = []
        stress_coop = []
        stress_resource = []
        stress_disharmony = []
        for d in range(1, max_dim):
            _, mds, stress = compute_mds(df, compute_joint=True,
                    columns=cols,
                    dim=d, return_stress=True)
            # corr_with_max_dim.append(corr_between_coords(mds_max, mds))
            stress_all.append(stress)
            _, mds, stress = compute_mds(df, compute_joint=True,
                    columns=reduced_cooperative,
                    dim=d, return_stress=True)
            stress_coop.append(stress)
            _, mds, stress = compute_mds(df, compute_joint=True,
                    columns=reduced_resource,
                    dim=d, return_stress=True)
            stress_resource.append(stress)
            _, mds, stress = compute_mds(df, compute_joint=True,
                    columns=reduced_disharmony,
                    dim=d, return_stress=True)
            stress_disharmony.append(stress)

        plt.plot(range(1, max_dim), stress_all, label="Stress All")
        plt.plot(range(1, max_dim), stress_coop, label="Stress Cooperative")
        plt.plot(range(1, max_dim), stress_resource, label="Stress Resources")
        plt.plot(range(1, max_dim), stress_disharmony, label="Stress Disharmony")
        plt.xlabel("Dimension")
        plt.xticks([2,4,6,8,10,12,14,16,18])
        plt.ylabel("Stress")
        plt.legend(loc="best")
        if save_file:
            plt.savefig(fname, bbox_inches="tight")
            print("Saved scree plot as {}".format(fname))
        else:
            plt.show()

    # Make a clustering of the results
    if False:
        _, mds, subaks = compute_mds(df, compute_joint=True, columns=cols,
                dim=19,
                get_names=True)
        kmeans = KMeans(n_clusters=5).fit_predict(mds)
        clusters = pd.DataFrame(kmeans, index=subaks, columns=['Clusters'])
        print(clusters.sort_values(by='Clusters'))

    if False:
        # mds = align_coords(mds)
        mds_joint = align_coords(mds_joint)

        # plt.subplot(121)
        # plt.scatter(mds[:,0], mds[:,1], c=mds[:,2], label='MDS', s=100)
        # ax1 = plt.gca()
        # plt.title("MDS")
        # plt.colorbar()

        # plt.subplot(122)
        plt.subplot(221)
        plt.scatter(mds_joint[:,0], mds_joint[:,1], label='MDS_Joint', s=90, 
                c=range(len(subaks)))
        plt.xlabel("x")
        plt.ylabel("y")
        ax1 = plt.gca()

        plt.subplot(223)
        plt.scatter(mds_joint[:,0], mds_joint[:,2], label='MDS_Joint', s=90, 
                c=range(len(subaks)))
        plt.xlabel("x")
        plt.ylabel("z")
        ax2 = plt.gca()

        plt.subplot(222)
        plt.scatter(mds_joint[:,2], mds_joint[:,1], label='MDS_Joint', s=90, 
                c=range(len(subaks)))
        plt.xlabel("z")
        plt.ylabel("y")
        ax3 = plt.gca()

        for i in range(len(subaks)):
            ax1.annotate(subaks[i], xy=(mds_joint[i, 0], mds_joint[i, 1]))
            ax2.annotate(subaks[i], xy=(mds_joint[i, 0], mds_joint[i, 2]))
            ax3.annotate(subaks[i], xy=(mds_joint[i, 2], mds_joint[i, 1]))
            
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(mds_joint[:,0], mds_joint[:,1], mds_joint[:,2], s=90, 
                c=range(len(subaks)))
        labels = []
        for i in range(len(subaks)):
            x, y, _ = proj3d.proj_transform(mds_joint[i, 0], mds_joint[i, 1], 
                    mds_joint[i, 2], ax.get_proj())
            label = ax.annotate(subaks[i], xy=(x, y), textcoords='offset points')
            labels.append(label)

        fig.canvas.mpl_connect('motion_notify_event', lambda e: update_position(e, 
            fig, ax, subaks, mds_joint, labels))

        plt.show()


    # print(columns)
    # columns = [columns[6]]
    # columns = ['pest_frequency_23']

    #### Most recently commented out
    # corrs = []
    # dims = [1, 2,3,4,5,6]
    # dims = list(range(1, 19))
    # for i in dims:
        # coords, subaks = get_mds(df, columns, dim=i, get_fisher=True)
        # corr = compute_correlation_with_mds(coords, get_hellinger_dist_mat(df, columns, get_fisher=True))
        # corrs.append(corr)
        # print("Correlation for dim=%d is %.3f" % (i, corr))

    # import matplotlib.pyplot as plt
    # plt.plot(dims, corrs)
    # plt.title("Correlation coefficient as function of dimension")
    # plt.xlabel("Dim")
    # plt.ylabel("Corr")
    # plt.show()
    #####

    # get_distribution_k_means(df, columns[-7:], k=4)

    # s1 = "Betuas, Keramas"
    # s2 = "Selukat"
    # reps = [10000, 20000, 50000, 100000, 150000, 200000]
    # ps = []
    # for rep in reps:
        # start = time.time()
        # H, perm, p = get_p_value(df, columns, s1, s2, repetitions=rep)
        # ps.append(p)
        # print("p = %.2e\t rep = %d\t time = %.2f secs" % (p, rep, time.time()-start))

    # import matplotlib.pyplot as plt

    # plt.plot(reps, ps)
    # plt.title("p value as function of repetitions")
    # plt.show()

