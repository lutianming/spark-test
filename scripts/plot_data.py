#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def plot(data):
    pos = data[:, 0] > 0
    neg = data[:, 0] <= 0
    plt.plot(data[pos, 1], data[pos, 2], 'r+')
    plt.plot(data[neg, 1], data[neg, 2], 'bo')
    plt.show()

if __name__ == '__main__':
    filename = argv[1]
    data = np.genfromtxt(filename, dtype=float, delimiter=" ")
    plot(data)
