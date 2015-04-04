#!/usr/bin/env python
from sys import argv
import numpy as np

def datagen(name, size, sigma=1):
    name = name.lower()
    X = None
    labels = None
    if(name == 'gaussian'):
        n = size/2
        x1 = sigma*np.random.normal(size=(n, 3))
        x1 = x1 + 1
        x2 = sigma*np.random.normal(size=(n, 3))
        x2 = x2 - 1
        X = np.vstack((x1, x2))
        labels = np.concatenate((np.ones(n), np.zeros(n)))

    elif(name == 'ball'):
        n = size/2
        X1 = sigma*np.random.normal(size=(n, 3))

        dim = 1.5
        x2 = np.linspace(-dim, dim, np.sqrt(n))
        y2 = x2
        x2v, y2v = np.meshgrid(x2, y2)
        z2v = dim**2 - x2v**2 - y2v**2

        n2 = x2v.size
        X2 = np.hstack((x2v.reshape((n2, 1)), y2v.reshape((n2, 1)), z2v.reshape((n2, 1))))
        # noise = np.random.uniform(-0.1, 0.1, size=(n2, 3))
        # X2 += noise
        X = np.vstack((X1, X2))
        labels = np.concatenate((np.ones(n), np.zeros(n2)))

    elif(name == 'checkers'):
        n = size/16

        for i in range(-2, 2):
            for j in range(-2, 2):
                x = i + np.random.rand(n)
                y = j + np.random.rand(n)
                points = np.vstack((x, y)).transpose()

                if X is None:
                    X = points
                else:
                    X = np.concatenate((X, points))

                l = (2*((i+j+4) % 2)-1)*np.ones(n)
                if labels is None:
                    labels = l
                else:
                    labels = np.append(labels, l)
                labels[labels <= 0] = 0
    elif(name == 'clowns'):
        n = size/2
        x1 = 6 * np.random.rand(n) - 3
        x2 = x1 ** 2 + np.random.randn(n)
        x = np.vstack((x1, x2)).transpose()
        x0 = sigma * np.random.randn(n, 2)
        x0[:, 1] += 6

        X = np.vstack((x, x0))
        X = (X - np.ones((2*n, 1))*np.mean(X, axis=0)).dot(np.diag(1 / np.std(X, axis=0)))
        labels = np.concatenate((np.ones(n), np.zeros(n)))
    else:
        pass
    return X, labels


if __name__ == '__main__':
    name = argv[1]
    size = int(argv[2])
    sigma = float(argv[3])
    filename = argv[4]

    X, labels = datagen(name, size, sigma)
    with open(filename, mode='w') as f:
        for label, x in zip(labels, X):
            line = "{0} {1} {2} {3}\n".format(label, x[0], x[1], x[2])
            f.write(line)
