#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
import math

def plot(data, surface=None):
    pos = data[:, 0] > 0
    neg = data[:, 0] <= 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(data[pos, 1], data[pos, 2], data[pos, 3], c='r', marker='o')
    ax.scatter(data[neg, 1], data[neg, 2], data[neg, 3], c='b', marker='x')

    if(surface is not None):
        X = surface[:, 0]
        Y = surface[:, 1]
        Z = surface[:, 2]

        n = int(math.sqrt(X.size))
        shape = (n, n)
        X = X.reshape(shape)
        Y = Y.reshape(shape)
        Z = Z.reshape(shape)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        # ax.contour(X, Y, Z)
    plt.show()

if __name__ == '__main__':
    filename = argv[1]
    data = np.genfromtxt(filename, dtype=float, delimiter=" ")

    if(len(argv) > 2):
        surface_file = argv[2]
        surface = np.genfromtxt(surface_file, dtype=float, delimiter=" ")
    else:
        surface = None
    plot(data, surface)
