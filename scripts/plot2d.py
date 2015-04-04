#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
from sys import argv

def plot(data, surface=None, vectors=None):
    pos = data[:, 0] > 0
    neg = data[:, 0] <= 0
    plt.plot(data[pos, 1], data[pos, 2], 'r+')
    plt.plot(data[neg, 1], data[neg, 2], 'bx')

    if(surface is not None):
        X = surface[:, 0]
        Y = surface[:, 1]
        Z = surface[:, 2]

        n = int(math.sqrt(X.size))
        shape = (n, n)
        X = X.reshape(shape)
        Y = Y.reshape(shape)
        Z = Z.reshape(shape)
        cs = plt.contour(X, Y, Z, levels=[-1, 0, 1])
        plt.clabel(cs, inline=1, fontsize=10)
    if(vectors is not None):
        pos = vectors[:, 0] > 0
        neg = vectors[:, 0] <= 0
        Y = vectors[:, 1]
        plt.plot(vectors[pos, 1], vectors[pos, 2], 'r^')
        plt.plot(vectors[neg, 1], vectors[neg, 2], 'b^')
    plt.show()

if __name__ == '__main__':
    filename = argv[1]
    data = np.genfromtxt(filename, dtype=float, delimiter=" ")

    if(len(argv) > 2):
        surface_file = argv[2]
        surface = np.genfromtxt(surface_file, dtype=float, delimiter=" ")
    else:
        surface = None

    if(len(argv) > 3):
        vectors_file = argv[3]
        vectors = np.genfromtxt(vectors_file, dtype=float, delimiter=" ")
    else:
        vectors = None
    plot(data, surface, vectors)
