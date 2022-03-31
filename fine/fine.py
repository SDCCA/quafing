#!/usr/bin/env python3
# encoding: utf-8



# Math imports
import numpy as np
from scipy.integrate import quad
import pandas as pd

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
#from mpltools import style
plt.style.use("ggplot")

# Density Estimation
#from deft import deft_1d, deft_2d
from scipy.stats import gaussian_kde
from astroML.plotting import hist
from astroML.density_estimation import bayesian_blocks
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# System and tests
import sys
import os
import pytest
import warnings

# MDS
from sklearn import manifold

# For shortest path calculation and community detection
import networkx as nx
import community
import metis

# Utility functions which don't have to be in a class and can be tested
# separately.

indices = []

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    raise ValueError("Value not found")

def symmetric_disc_kl(p1, p2, base=None):
    """ Returns the symmetrized (using minimum) discrete KL-divergence.

    Args:
        p1 (@todo): @todo
        p2 (@todo): @todo
        base (@todo): @todo

    Returns: @todo

    """
    kl1 = discrete_kl(p1, p2, base)
    kl2 = discrete_kl(p2, p1, base)
    if np.isinf(kl1) and np.isinf(kl2):
        return np.inf
    m = 2.0 * np.nanmin([kl1, kl2])
    return np.sqrt(m)

def discrete_kl(p1, p2, base=None):
    """ Returns the KL divergence between two _discrete_ pdfs, defined as a
    dictionary with possible values as keys and probabilities as values.
    i.e. p1 and p2 are the result of a call to DiscreteEstimator.get_pdf().


        KL(p1||p2) = sum_i p1_i * log (p1_i/p2_i)
        Parameters
        ----------
        p1: dict
        The pdf to compute the kl divergence from.
        p2: dict
        The pdf to compute the kl divergence to.
        base: int or None
        None (natural), 2 or 10 base for the logarithm

    returns: The KL-divergence computed.

    """
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)
    # TODO: Fix this
    #assert np.sum(p1.values()) == 1.0
    #assert np.sum(p2.values()) == 1.0
    assert base in (None, 2, 10)

    if base is None:
        log = np.log
    elif base == 2:
        log = np.log2
    else:
        log = np.log10

    v1, v2 = list(p1.keys()), list(p2.keys())
    kl = 0.0
    for k in v1:
        if k in v2:
            if p1[k] == 0:
                continue
            kl += p1[k] * (log(p1[k]) - log(p2[k]))
        else:
            warnings.warn("Support of p2 smaller than p1 in discrete_kl", RuntimeWarning)
            # TODO
            return np.inf
            #kl += 0
    return kl

def continuous_kl(p1, p2, bbox=(-np.inf, np.inf), base=None):
    """ Computes the continuous KL divergence from p1 to p2, limited to the
    bounding box bbox.

    Args:
        p1 (function): Distribution from
        p2 (function): Distribution to
        bbox (2-tuple): Tuple with integration boundaries.
        base (None, 2 or 10): The base of the logarithm (natural, 2 or 10).

    Returns: The computed KL divergence.

    """
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 2 and bbox[0] < bbox[1]
    assert hasattr(p1, '__call__')
    assert hasattr(p2, '__call__')
    assert base in (None, 2, 10)

    if base is None:
        log = np.log
    elif base == 2:
        log = np.log2
    else:
        log = np.log10

    def kl(x, p1, p2):
        if p1(x) == 0.0:
            return 0.0
        if p2(x) == 0.0:
            warnings.warn("Continuous KL has support for p2 where p1 is zero.", RuntimeWarning)
            # TODO
            return np.inf
            #return 0.0
        return p1(x) * (log(p1(x)) - log(p2(x)))

    return quad(lambda x: kl(x, p1, p2), bbox[0], bbox[1])[0]


def discrete_hellinger(p1, p2):
    """ Computes the Hellinger distance between p1 and p2.

    Args:
        p1 : dict with keys as values and values as probabilities
        p2 : dict with keys as values and values as probabilities

    Returns: The computed Hellinger distance.

    """
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)
    #assert np.sum(p1.values()) == 1.0, "Sum %f not equal to one!" % np.sum(p1.values())
    #assert np.sum(p2.values()) == 1.0, "Sum %f not equal to one!" % np.sum(p2.values())

    k1 = list(p1.keys())
    k2 = list(p2.keys())

    tot_sum = 0.0
    for k in k1:
        if k in k2:
            tot_sum += (np.sqrt(p1[k]) - np.sqrt(p2[k])) ** 2
        else:
            tot_sum += p1[k]
    non_k1 = [p for p in k2 if p not in k1]
    for k in non_k1:
        tot_sum += p2[k]

    return 2 * np.sqrt(tot_sum)

def continuous_hellinger(p1, p2, bbox=(-np.inf, np.inf)):
    """ Computes the Hellinger distance of two continuous PDFs.

    Args:
        p1 (function): PDF1 to compute Hellinger distance.
        p2 (function): PDF2 to compute Hellinger distance.
        bbox (2-tuple): Integration bounds.

    Returns: The computed Hellinger distance.

    """
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 2 and bbox[0] < bbox[1]
    assert hasattr(p1, '__call__')
    assert hasattr(p2, '__call__')

    def hell(x, p1, p2):
        return (np.sqrt(p1(x)) - np.sqrt(p2(x))) ** 2

    integral = quad(lambda x: hell(x,p1,p2), bbox[0], bbox[1])
    return np.sqrt(integral[0])


class DensityEstimator():

    """ A class for performing the nonparametric density estimation on the data.
        Intended for both discrete and continuous densities.
    """

    def __init__(self, data, discrete=True, bins=None, col_metadata=None):
        """ Initializes the DensityEstimator.

        Args:
            data (1D array of data points): The data whose density is to be evaluated.

        Kwargs:
            discrete (Boolean, True): Whether the samples come from an inherently discrete density.
            bins (None or array of edges): If supplied and discrete=False, gives the (adaptive) binning
                                           to be used to get a discrete estimate of the continuous density.

        """

        self._data = data
        self._discrete = discrete
        self._bins = bins
        self._col_metadata = col_metadata

        if discrete:
            if isinstance(data, pd.Series):
                self._unique = data.unique()
            else:
                self._unique = np.array(list(set(data)))
            disc_pdf = {}
            for val in self._unique:
                count = np.sum(self._data == val)
                disc_pdf[val] = count / len(self._data)
            self._discrete_pdf = disc_pdf
        else:
            if bins is not None:
                # Compute the probabilities of each bin
                h = np.histogram(data, bins=bins, normed=True)[0]
                widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
                probs = h * widths
                # The 'value' of each key is the bin number
                disc_pdf = {}
                for i in range(len(bins)-1):
                    disc_pdf[i] = probs[i]
                self._discrete_pdf = disc_pdf

    def is_discrete(self):
        """ Is the data inherently discrete?
        Returns: @todo

        """
        return self._discrete

    def get_discrete_pdf(self):
        """ Returns a discrete pdf (if available). If none is computed raises
        an error.
        Returns: @todo

        """
        if hasattr(self, "_discrete_pdf"):
            return self._discrete_pdf
        else:
            raise Exception("No Discrete PDF was calculated! Use bins before calling this method.")

    def get_continuous_pdf(self, method="kde"):
        """
        Returns a nonparametric estimation of the PDF data.
        """
        assert method in ["kde", "deft"]

        if method == "kde":
            try:
                return gaussian_kde(self._data, bw_method="silverman")
            except:
                print(self._data)
                raise Exception("stop!!")
        elif method == "deft":
            bbox = self.get_bbox()
            G = 400
            alpha=3
            return deft_1d(self._data, bbox=bbox, G=G, alpha=alpha)

    def get_bbox(self):
        """
        Computes a bounding box estimate for the DEFT pdf approximation.
        """
        return (np.min(self._data) * 0.5, np.max(self._data) * 1.5)

    def get_metadata(self):
        return self._col_metadata


class MultiDimensionalPDF:
    def __init__(self, data, metadata, bins=None):
        """
        Initializes the MultiDimensionalPDF.

        Parameters
        ----------
        data : A list of data lists corresponding to the different questions.

        metadata : Description of the column data types

        bins (None or list of list of edges): If not None, a list of edges to determine the binning of the continuous variables.

        """
        assert isinstance(data, pd.DataFrame)
        assert len(data.columns) == len(metadata)

        if bins is not None:
            assert len(bins) == np.sum([1 for c in metadata if c['type'] == 'c'])

        self._data = data
        self._meta = metadata
        self._bins = bins

        # Perform an estimate for each column of the data using each description
        pdfs = []
        count_cont = 0
        for i in range(len(metadata)):
            if metadata[i]['type'] in ('g', 'e'):
                continue
            b = None
            if metadata[i]['type'] == 'c':
                if bins is not None:
                    b = bins[count_cont]
                    count_cont += 1

            d = data[metadata[i]['name']]

            pdfs.append(DensityEstimator(d, discrete=metadata[i]['type'] not in ('c','g'), bins=b, col_metadata=metadata[i]))
        self._pdfs = pdfs

    def get_pdfs(self):
        """
        Returns a list of the estimated pdfs
        """
        return self._pdfs

    def get_metadata(self):
        """
        Returns the column metadata.
        """
        return self._meta

class InformationDistance:

    def __init__(self, p1, p2):
        """
        p1 and p2 are the two MultiDimensionalPDFs to use for the distance
        """
        #assert isinstance(p1, MultiDimensionalPDF)
        #assert isinstance(p2, MultiDimensionalPDF)
        assert len(p1.get_pdfs()) == len(p2.get_pdfs())
        assert p1.get_metadata() == p2.get_metadata()

        self._p1 = p1
        self._p2 = p2

    def get_distance(self, method="hellinger", dist_type="rms", use_binning=False, deft=False):
        """
        Computes the distance between two MultiDimensionalPDFs.
        The computation method might be either Hellinger or Symmetric KL and
        the distance can either be an average of the distances for the individual
        questions or the total sum of distances.

        Parameters
        ----------
        method (str) : Either hellinger or kl distance
        dist_type (str) : Either 'rms', 'sum' or 'average'
        use_binning (bool) : Use the bins computed for the continuous PDFs?
        deft (bool) : Use DEFT to do the continuous density estimation?

        Returns (float) : The computed distance
        """
        assert isinstance(method, str)
        assert method in ("hellinger", "kl")
        assert isinstance(dist_type, str)
        assert dist_type in ("rms", "sum", "average")

        pdfs1 = self._p1.get_pdfs()
        pdfs2 = self._p2.get_pdfs()

        if method == "hellinger":
            cont_dist = continuous_hellinger
            disc_dist = discrete_hellinger
        elif method == "kl":
            cont_dist = continuous_kl
            #disc_dist = discrete_kl
            disc_dist = symmetric_disc_kl

        distances = []

        metadata = self._p1.get_metadata()
        for i in range(len(pdfs1)):
            m = pdfs1[i].get_metadata()
            if m['type'] in ('g', 'e'):
                continue
            if m['type'] != 'c' or use_binning:
                dist = disc_dist(pdfs1[i].get_discrete_pdf(), pdfs2[i].get_discrete_pdf())
                distances.append(dist)
                #distances.append(disc_dist(pdfs1[i].get_discrete_pdf(), pdfs2[i].get_discrete_pdf()))
            else:
                distances.append(cont_dist(pdfs1[i].get_continuous_pdf(), pdfs2[i].get_continuous_pdf()))

        distances = np.array(distances)
        global indices
        if np.any(np.isinf(distances)):
            for i in range(len(distances)):
                if np.isinf(distances[i]):
                    indices.append(i)

        #count = 0
        #for i in range(len(pdfs1)):
            #m = pdfs1[i].get_metadata()
            #print("Name: %s" % m['name'])
            #print("value: %.2f" % distances[count])
            #count += 1

        if dist_type == "rms":
            return np.sqrt(np.sum(distances ** 2))
        elif dist_type == "sum":
            return np.array(distances).sum()
        return np.array(distances).average()

    def get_min_distance(self):
        """ Returns the minimal distance between the two MultiDimensionalPDFs
        by computing all distance measures and getting the minimum.

        Returns: The minimal distance.

        """
        hd = self.get_distance(method="hellinger", dist_type="sum")
        kl = self.get_distance(method="kl", dist_type="sum")
        return min(hd, kl)

class FINE:

    """ Runs the FINE algorithm on data. """

    def __init__(self, filename, type_row=1, question_number_row=2, header_row=3,
                        name_row=3, start_row=4, group_by=0, skip_footer=0, sheet_number=1,
                        shuffle=False):
        """ Initialize the FINE object with data loaded from the file specified
            by filename.

            Each column in the data should have these attributes (given as rows):
                index: the index of the column
                type: one of 'egcuob' with e: excluded
                                           g: group by this column
                                           c: continuous variable
                                           u: unordered discrete
                                           o: ordered discrete
                                           b: binary
                name: human readable text to identify the question
                question_number: what was the question number in the questionnaire.

        Args:
            filename (string): Name of excel file with the data.

        Kwargs:
            type_row (int): Row number of column type definition.
            question_number_row (int): Row where the question number is stored.
            name_row (int) : Row where the column headers are located.
            start_row (int) : Where does the actual data begin.
            group_by (int) : Which column to use for grouping.
            skip_footer (int): Are there rows at the end of file to ignore?
            header_row (int) : The row to use for headers.
            sheet_number (int) : The workbook number to use.
            permutation (list of int) : A random permutation of the results to use
                for the computation of the FINE algorithm.

        """
        assert isinstance(filename, str) and os.path.isfile(filename)
        self._filename = filename

        wb = pd.ExcelFile(filename)

        # Extract metadata from the excel sheet
        #types = wb.book.sheet_by_index(sheet_number).row(type_row)
        ws = wb.book.get_sheet_by_name(wb.book.sheetnames[sheet_number -1])
        types = ws[type_row]
        self._types = types = [t.value for t in types]
        #qns = wb.book.sheet_by_index(sheet_number).row(question_number_row)
        qns = ws[question_number_row]
        qns = [t.value for t in qns]
        question_numbers = []
        for t in qns:
            if not isinstance(t, str):
                try:
                    t = str(int(t))
                except:
                    t = t
            question_numbers.append(t)
        self._question_numbers = question_numbers
        #column_names = [t.value for t in wb.book.sheet_by_index(sheet_number).row(name_row)]
        column_names = [t.value for t in ws[name_row]]

        # Compute column indices to include and create a
        # metadata structure for column types.
        parse_cols = []
        col_metadata = []
        for i in range(len(types)):
            if types[i] != 'e':
                parse_cols.append(i)
                col_metadata.append({'type' : types[i],
                                     'name' : column_names[i],
                                     'number' : question_numbers[i]
                                     })

        self._column_metadata = col_metadata
        #raw_data = wb.parse(sheetname=sheet_number, header=header_row,
                            #skip_footer=skip_footer, parse_cols=parse_cols, skiprows=skip_rows)
        raw_data = wb.parse(sheetname=sheet_number, header=start_row-2, parse_cols=parse_cols)

        self._raw_data = raw_data

        cols = raw_data.columns

        if shuffle:
            subak_assignment = raw_data.loc[:,cols[group_by]].copy()
            np.random.shuffle(subak_assignment)
            raw_data.loc[:, cols[group_by]] = subak_assignment
        grouped = raw_data.groupby(by=cols[group_by])

        col_data = grouped[cols[group_by]].unique()
        self._col_data = col_data
        groups = []
        for i in range(len(col_data)):
            groups.append(raw_data[raw_data[cols[group_by]] == col_data.iloc[i][0]])
        self._groups = groups

        s = "".join([cm['type'] for cm in col_metadata])
        s = s.replace("g", "").replace("b", "u")

        ## Compute the bayesian binning for data from all groups together.
        #cont_data = data.iloc[:, 1:].loc[:, np.array(desc) == 1]
        cont_cols = [c['name'] for c in col_metadata if c['type'] == 'c']
        cont_data = raw_data[cont_cols]

        bins_list = []
        for i in range(len(cont_data.columns)):
            d = cont_data.iloc[:,i]
            bb = bayesian_blocks(d)
            bins_list.append(bb)

        self._bins_list = bins_list

        # Compute the MultiDimensionalPDFs for each group
        ps = []
        for i in range(len(groups)):
            p = MultiDimensionalPDF(groups[i], col_metadata, bins_list)
            ps.append(p)

        self._pdfs = ps

    def get_distance_matrix(self, method="hellinger", dist_type="rms",
                            use_binning=False, to_shortest_path=True):
        """ Computes the distances matrix

        Kwargs:
            method ("hellinger", "kl") : Which method to use to estimate distances?
            dist_type ("rms", "sum", "average") : How to collect all individual distances.
            use_binning (bool, False) : Use binning for continuous variables?
            to_shortest_path (bool, True) : Run through the Dijkstra algorithm to obtain shortest paths?

        """
        data = self._groups
        distances = np.zeros((len(data), len(data)))
        ps = self._pdfs
        for i in range(len(data)):
            for j in range(i):
                if i == j: continue
                ifd = InformationDistance(ps[i], ps[j])
                distances[i, j] = ifd.get_distance(method=method, dist_type=dist_type, use_binning=use_binning)
                distances[j, i] = distances[i, j]

        if to_shortest_path:
            return self.get_shortest_path_matrix(distances)
        return distances

    def get_shortest_path_matrix(self, A):
        """ Takes the distance matrix and computes the shortest path matrix
        between all pairs of units in the groups.

        Args:
            A (numpy square matrix): A distances matrix (all entries positive)

        Returns: A matrix with shortest distances.

        """
        assert isinstance(A, np.ndarray)
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
        assert np.all(A >= 0)

        g = nx.from_numpy_matrix(A)
        paths = dict(nx.all_pairs_dijkstra_path_length(g))
        new_mat = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(i):
                new_mat[i, j] = new_mat[j, i] = paths[i][j]

        return new_mat

    def get_mds(self, d=2, method="hellinger", dist_type="rms", use_binning=False, return_stress=False, return_all=False):
        """ Returns the Multidimensional Scaling embedding.
        Args:
            d (int): The dimensionality of the MDS embedding.
        Returns: The d-dimensional embedding.

        """
        assert isinstance(d, int)
        assert d > 1, "Dimension of MDS computation must be greater than 1"
        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=d, metric=True, max_iter=3000, eps=1e-9,
                           random_state=seed, dissimilarity="precomputed",
                           n_jobs=1)
        dist_matrix = self.get_distance_matrix(method=method, dist_type=dist_type, use_binning=use_binning, to_shortest_path=True)
        assert np.all(np.isfinite(dist_matrix)), "All distances in the distance matrix must be finite!"
        fit = mds.fit(dist_matrix)
        embedding = fit.embedding_
        stress = fit.stress_
        if return_all:
            return embedding, dist_matrix, stress
        if return_stress:
            return stress
        return embedding, dist_matrix

    def plot_stress(self, method="hellinger"):
        """ Plots the stress as a function of the dimension, to find the "true"
        dimensionality of the data
        Returns: Nothing

        """
        max_dim = len(self._groups) - 1

        stresses = list()
        dims = list(range(2, max_dim + 1))
        for d in dims:
            stresses.append(self.get_mds(d=d, method=method, use_binning=True, return_stress=True))
        plt.plot(dims, stresses)
        plt.xlabel("Dimension")
        plt.ylabel("Stress")
        plt.title("MDS computed stress as function of dimension")
        plt.show()

   
    def plot_embedding(self, d=2, method="hellinger", dist_type="rms", use_binning=False,
                       show_labels=True, show=True, color="distance"):
        """ Plots the embedding computed in d dimensions (d=2,3)

        Kwargs:
            d (2 or 3): The dimension to use for the plotting.
            color (str): What to use to color the nodes?
                         "distance": according to distance from center of mass
                         "partition": according to network partitioning

        """
        assert isinstance(d, int)
        #assert d in (2, 3, 4)
        assert color in ("distance", "partition", "metis"), "color must be one of 'distance', 'partition', 'metis'."

        em, dist_matrix = self.get_mds(d=d, method=method, dist_type=dist_type, use_binning=use_binning)

        G = nx.from_numpy_matrix(dist_matrix)
        part = community.best_partition(G)
        part2 = metis.part_graph(G)

        if d == 2:
            x, y = em[:,0], em[:,1]
            if color == "distance":
                cms = (np.average(em[:,0]), np.average(em[:,1]))
                dx = x - cms[0]
                dy = y - cms[1]
                t = np.sqrt(dx**2 + dy**2)
                t = (t - t.min()) * 100.0 / (t.max() - t.min())
            elif color == "partition":
                t = [part[i] for i in range(len(list(part.keys())))]
            else: # metis
                t = part2[1]

            fig, ax = plt.subplots()
            ax.scatter(x, y, c=t, cmap=cm.jet)
            if show_labels:
                labels = self.get_labels()
                for i, txt in enumerate(labels):
                        ax.annotate(txt[0], (x[i]+0.05, y[i]))
        elif d == 3:
            if color == "distance":
                cms = (np.average(em[:,0]), np.average(em[:,1]), np.average(em[:,2]))
                dx = em[:,0] - cms[0]
                dy = em[:,1] - cms[1]
                dz = em[:,2] - cms[2]
                t = np.sqrt(dx**2 + dy**2 + dz**2)
                t = (t - t.min()) * 100.0 / (t.max() - t.min())
            elif color == "partition":
                t = [part[i] for i in range(len(list(part.keys())))]
            else: # metis
                t = part2[1]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(em[:,0], em[:,1], em[:,2], marker='o', s=200, c=t, cmap=cm.jet)
            if show_labels:
                labels = self.get_labels()
                global lbls
                lbls = []
                for i, txt in enumerate(labels):
                    x2, y2, _ = proj3d.proj_transform(em[i,0], em[i,1],em[i,2], ax.get_proj())
                    lbls += [ax.annotate(txt[0], (x2+0.002, y2))]

                #def update_position(e):
                    #global lbls
                    #print("update_position")
                    #for i, txt in enumerate(labels):
                        #x2, y2, _ = proj3d.proj_transform(em[i,0], em[i,1],em[i,2], ax.get_proj())
                        #lbls[i].xy = (x2, y2)
                        #lbls[i].update_positions(fig.canvas.renderer)
                    #fig.canvas.draw()
                #fig.canvas.mpl_connect('button_release_event', update_position)
        elif d > 3:
            pass
        plt.title("Subak positions computed using %s method" % method)
        plt.show()
    
    def get_labels(self):
        """ Returns the labels of the grouping
        Returns: @todo

        """
        return self._col_data


##### UNIT TEST AREA #####

def test_kl_distance_infinite_when_support_of_2nd_pdf_smaller_than_first():
    p1 = {1 : 0.2, 2 : 0.1, 3 : 0.7}
    p2 = {1 : 0.2, 2 : 0.1, 4 : 0.7}
    assert discrete_kl(p1, p2) == np.inf

def test_kl_distance_zero_if_equal():
    p1 = {1 : 0.2, 2 : 0.1, 3 : 0.7}
    p2 = {1 : 0.2, 2 : 0.1, 3 : 0.7}
    assert discrete_kl(p1, p2) == 0.0

def test_hellinger_distance_zero_if_equal():
    p1 = {1 : 0.2, 2 : 0.1, 3 : 0.7}
    p2 = {1 : 0.2, 2 : 0.1, 3 : 0.7}
    assert discrete_hellinger(p1, p2) == 0.0

def test_hellinger_distance_is_symmetric():
    p1 = {1 : 0.2, 2 : 0.1, 3 : 0.3, 4 : 0.4}
    p2 = {1 : 0.2, 2 : 0.1, 4 : 0.7}
    assert discrete_hellinger(p1, p2) == discrete_hellinger(p2, p1)

def test_continuous_kl_converges_to_analytic_result():
    from scipy.stats import norm
    s1 = 1.0
    s2 = 2.0
    m1 = 1
    m2 = 1
    p1 = norm.freeze(loc=m1, scale=s1).pdf
    p2 = norm.freeze(loc=m2, scale=s2).pdf
    kl_dist = continuous_kl(p1, p2, base=None)
    analytic = np.log(s2/s1) + (s1**2 + (m1-m2)**2) / (2 * (s2)**2) - 0.5
    # On my machine relative error around 1e-16
    assert (np.abs(kl_dist - analytic) / analytic) < 1e-4

def test_continuous_hellinger_converges_to_analytic_result():
    from scipy.stats import norm
    s1 = 1.0
    s2 = 2.0
    m1 = 1
    m2 = 1
    p1 = norm.freeze(loc=m1, scale=s1).pdf
    p2 = norm.freeze(loc=m2, scale=s2).pdf
    hell_dist = continuous_hellinger(p1, p2)
    analytic = 1 - np.sqrt((2*s1*s2)/(s1**2+s2**2)) * np.exp(-0.25*((m1-m2)**2)/(s1**2+s2**2))
    analytic = np.sqrt(2 * analytic)
    # On my machine relative error around 1e-16
    assert (np.abs(hell_dist - analytic) / analytic) < 1e-4

if __name__ == "__main__":
    f = FINE("/Users/eslt0101/Projects/SABM/testpy2/omri_subak_data.xlsx")
    #f = FINE("/home/omri/repos/research/subaks/in_sg/data/omri_subak_data.xlsx")
    # f = FINE("/home/omri/repos/research/subaks/in_sg/data/shortest.xlsx")
    #print(f.get_distance_matrix(method='kl', use_binning=False))
    #print(indices)
    #f.plot_embedding(d=2, method='kl', use_binning=True, show_labels=True, color="distance")
    f.plot_embedding(d=2, method='hellinger', use_binning=True, show_labels=True, color="distance")
    #f.plot_embedding(d=4, method='kl', use_binning=True, show_labels=True, color="distance")
    #f.plot_embedding(d=5, method='kl', use_binning=True, show_labels=True, color="distance")
    #f.plot_embedding(d=6, method='kl', use_binning=True, show_labels=True, color="distance")
    #plt.hist(indices, bins=28)
    #plt.gcf().set_size_inches(18, 10)
    #plt.savefig("3d_plot.png")
    #plt.show()
    #f.plot_stress(method="hellinger")
    #plt.show()
