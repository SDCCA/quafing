#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy import sin, cos, linspace, pi, array, zeros, ones
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D

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

theta = 20 * pi / 180
alpha = 45 * pi / 180
beta = 45 * pi / 180

plot_sphere = True
save_file = False
filename = "../paper/images/sphere3.pdf"

t = np.linspace(0, 1, 1000)

if plot_sphere:
    dim = 3
    m = 2
    count = 1000

    # theta_line = linspace(0, pi/2, 1000)
    # x_line = linspace(0, 1, 1000)
    # y_line = -4 * (x_line**2 - x_line)
    # phi_line = linspace(0, pi/2, 1000)
    # phi_line = pi/2 * y_line
    # phi_line = pi/2 * sin(linspace(0, 2*pi, 1000))**2

    angles = zeros(shape=(dim - 1, count))
    # angles[0, :] = theta_line
    # angles[1, :] = phi_line
    angles[0, :] = (pi/2) * sin(t * pi * m)**2
    angles[1, :] = linspace(0, pi/2, count)

    line_coords = get_coords(angles)
    xp_line = line_coords[2,::50]
    yp_line = line_coords[0,::50]
    zp_line = line_coords[1,::50]

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    ax.view_init(elev=26, azim=40)

    ax.plot(line_coords[2], line_coords[0], line_coords[1], color="magenta",
            lw=4)
    ax.scatter(xp_line, yp_line, zp_line, color="green", s=400)

    # Draw the sphere
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s=np.cos(u)*np.sin(v)
    y_s=np.sin(u)*np.sin(v)
    z_s=np.cos(v)
    ax.plot_wireframe(x_s, y_s, z_s, color="gray", alpha=.6)

    # Draw the coordinate system
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    x_axis = Arrow3D([0,1],[0,0],[0,0], mutation_scale=20, lw=4, arrowstyle="-|>", color="k")
    ax.add_artist(x_axis)

    y_axis = Arrow3D([0,0],[0,1],[0,0], mutation_scale=20, lw=4, arrowstyle="-|>", color="k")
    ax.add_artist(y_axis)

    z_axis = Arrow3D([0,0],[0,0],[0,1], mutation_scale=20, lw=4, arrowstyle="-|>", color="k")
    ax.add_artist(z_axis)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.8, 0.8)
    plt.axis("off")
    if save_file:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
