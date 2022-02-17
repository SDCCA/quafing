"""
Reproduce Figure 2 from the following paper:

    Omri Har-Shemesh et al. Scientific Reports 10, 8633 (2020)
    https://doi.org/10.1038/s41598-020-63760-8

This script is based on `spherical.py`.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from hypersphere import get_curve_samples


if __name__ == "__main__":
    Axes3D

    filename = "reproduce-figure-2.pdf"

    # dim = number_a ** number_q = 3
    number_q = 1
    number_a = 3

    samples = 20  # number of groups
    count = 1000  # sample groups from this number of points on the curve

    # parameters defining the curves on the manifold
    ms = [1, 2, 2]
    kappas = [1, 1, 2]

    n_plots = len(ms)
    assert len(ms) == len(kappas)

    fig = plt.figure(figsize=(n_plots * 5, 5))

    for n in range(n_plots):

        m = ms[n]
        kappa = kappas[n]

        sample_probs = get_curve_samples(number_q=number_q, number_a=number_a,
                                         samples=samples, count=count,
                                         m=m, sin_angle=kappa-1)
        line_probs = get_curve_samples(number_q=number_q, number_a=number_a,
                                       samples=count, count=count,
                                       m=m, sin_angle=kappa-1)

        ax = fig.add_subplot(1, n_plots, n + 1, projection="3d")
        ax.set_aspect("equal")
        ax.view_init(elev=26, azim=40)

        ax.plot(line_probs[:, 2], line_probs[:, 0], line_probs[:, 1],
                color="magenta", lw=1)
        ax.scatter(sample_probs[:, 2], sample_probs[:, 0], sample_probs[:, 1],
                   color="green", s=40)

        # Draw the sphere
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x_s = np.cos(u)*np.sin(v)
        y_s = np.sin(u)*np.sin(v)
        z_s = np.cos(v)
        ax.plot_wireframe(x_s, y_s, z_s, color="gray", alpha=.6)

        plt.title("m = {}, k = {}".format(m, kappa))
        plt.axis("off")
        plt.tight_layout()

    plt.savefig(filename)
    plt.show()
