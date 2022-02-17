"""
Reproduce Figure 3 from the following paper:

    Omri Har-Shemesh et al. Scientific Reports 10, 8633 (2020)
    https://doi.org/10.1038/s41598-020-63760-8
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

from hypersphere import get_curve_samples, get_questionnaires, \
    multi_partite_distance, compute_mds, get_true_mds, align_pca_mds, \
    corr_between_coords


if __name__ == "__main__":

    filename = "reproduce-figure-3.pdf"

    number_q = 8  # number of questions
    number_a = 3  # number of answers
    m = 3  # sub-manifold parameter

    dim = 2  # dimensions of FINE output

    # parameter scan
    Ks = [20, 20, 20, 50, 50, 50]
    kappas = [1, 2, number_a**number_q - 1, 1, 2, number_a**number_q - 1]
    responses = [25, 25, 25, 50, 50, 50]

    n_sets = len(Ks)
    assert n_sets == len(kappas)
    assert n_sets == len(responses)

    plt.figure(figsize=(10, 5 * n_sets))

    for n_set in range(n_sets):
        K = Ks[n_set]
        kappa = kappas[n_set]
        count_answers = responses[n_set]

        probs_filename = "probs-K{:03d}-kappa{:04d}-responses{:03d}.npy"\
            .format(K, kappa, count_answers)
        if os.path.isfile(probs_filename):
            probs = np.load(probs_filename)
        else:
            probs = get_curve_samples(number_q=number_q, number_a=number_a,
                                      samples=K, m=m, sin_angle=kappa-1)
            np.save(probs_filename, probs)

        df = get_questionnaires(probs, count_answers=count_answers,
                                number_q=number_q, number_a=number_a)

        # calculate multipartite distance for coloring samples
        KLs = multi_partite_distance(probs)

        # get theoretical embedding
        true_mds = get_true_mds(probs)

        # FINE
        mds, mds_joint = compute_mds(df, dim=dim, compute_joint=True)

        # align the coordinates
        mds_joint = align_pca_mds(true_mds, mds_joint)

        # plot results
        plt.subplot(n_sets, 2, n_set*2 + 1)
        plt.scatter(true_mds[:, 0], true_mds[:, 1], label="True FI", c=KLs,
                    marker="o", s=100, zorder=3)
        plt.plot(true_mds[:, 0], true_mds[:, 1], label="True FI",
                 lw=3, zorder=2)
        plt.title("K = {}, k = {}".format(K, kappa))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.subplot(n_sets, 2, n_set*2 + 2)
        plt.scatter(mds_joint[:, 0], mds_joint[:, 1], marker="p",
                    label="FI Joint", c=KLs, s=100, zorder=3)
        plt.plot(mds_joint[:, 0], mds_joint[:, 1],  label="FI Joint", lw=3,
                 zorder=2)
        fi_corr = corr_between_coords(true_mds, mds_joint)
        plt.title("FI (Corr: %.3f)" % fi_corr)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()

    plt.savefig(filename)
    plt.show()
