#!/usr/bin/env python
from sys import argv
import numpy as np

def datagen(name, size, sigma=1):
    name = name.lower()
    X = None
    labels = None
    if(name == 'linear'):
        n = size/2
        X1 = sigma*np.random.normal(size=(n, 2))
        X1 = X1 + 1
        X2 = sigma*np.random.normal(size=(n, 2))
        X2 = X2 - 1
        X = np.vstack((X1, X2))
        labels = np.concatenate((np.ones(n), np.zeros(n)))

    elif(name == 'circle'):
        n = size/2
        X1 = sigma*np.random.normal(size=(n, 2))

        r = np.random.uniform(1, 1.5, size=(n, 1))
        angle = np.random.normal(size=(n, 1))*np.pi*2
        x2 = np.cos(angle)*r
        y2 = np.sin(angle)*r
        X2 = np.hstack((x2, y2))

        X = np.vstack((X1, X2))
        labels = np.concatenate((np.ones(n), np.zeros(n)))

    elif(name == 'mixture'):
        n = size/3
        x1 = sigma * np.random.randn(n) + 0.3
        x2 = sigma * np.random.randn(n) - 0.3
        x3 = sigma * np.random.randn(n) - 1
        y1 = sigma * np.random.randn(n) + 0.5
        y2 = sigma * np.random.randn(n) - 0.5
        y3 = sigma * np.random.randn(n) - 1
        x = np.concatenate((x1, x2, x3))
        y = np.concatenate((y1, y2, y3))
        labels = np.concatenate((np.ones(n), np.zeros(n), np.ones(n)))
        X = np.vstack((x, y)).transpose()

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
            line = "{0} {1} {2}\n".format(label, x[0], x[1])
            f.write(line)
