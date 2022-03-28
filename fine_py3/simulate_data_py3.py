#!/usr/bin/env python3
# encoding: utf-8
"""
Simulate data for sparse FINE
-----------------------------

This script simulates data for Subak analysis which should enable me to verify
different statistical properties of the results, test how sparsity affects
FINE, etc...

Written by
----------

Omri Har-Shemesh,
University of Amsterdam

"""



# Standard imports
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

import simplejson

from fine_py3 import *


## Parameters of the 'simulation'

# Number of parameters in the model
N_params = 5

# Number of questions in the questionnaire
N_questions = 12

# Number of answers per question
N_answers = 4

# Number of groups (Subaks)
N_groups = 10

# Number of of answers per group (fixed)
N_answers_per_group = 25

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

if __name__ == '__main__':
    fname = "one_parameter_with_kl"
    create_excel_file("sim_res/%s" % fname, N_questions=5, N_answers_per_group=30, N_groups=30, N_params=1, seed=100)
    f = FINE("sim_res/%s.xlsx" % fname, start_row=3)
    f.plot_stress(method="kl")
    f.plot_embedding(d=2, method="kl")
    f.plot_embedding(d=3, method="kl")
