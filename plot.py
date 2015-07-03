#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import groupby

def read_result(input):
    d = {}

    with open(input, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                if line[1] == '#':
                    continue
                name = line.split()[1]
                d[name] = ([], [])
            else:
                tokens = line.split()
                x = d[name][0]
                y = d[name][1]
                x.append(float(tokens[0]))
                y.append(float(tokens[1]))
    del d['step']
    return d

def read_loss(input):
    X = []
    Y = []
    with open(input, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue

            tokens = line.split()
            if len(tokens) < 2:
                continue

            x = int(tokens[0]) / 1000
            y = float(tokens[1])
            X.append(x)
            Y.append(y)
    return (X, Y)

def plot_loss(basedir, features):
    files = [f for f in os.listdir(basedir)]

    for (i, feature) in enumerate(features):
        plt.figure(i)

        inputs = [f for f in files if f.startswith(feature)]
        groups = {}
        func = lambda x: x.split("#")[-1]
        inputs = sorted(inputs, key=func)
        for k, g in groupby(inputs, func):
            groups[k] = list(g)

        fig, axs = plt.subplots(1, len(groups), sharey=True)
        for (ax, key) in zip(axs, groups.keys()):
            legends = []
            for v in groups[key]:
                path = os.path.join(basedir, v)

                x, y = read_loss(path)
                ax.plot(x[30:], y[30:])
                legends.append(v.split("#")[1])
            ax.legend(legends)
            ax.set_title(key.split(".")[0])

            ax.set_xlabel("time(second)")
            ax.set_ylabel("loss")
        fig.suptitle(feature)
    plt.show()

def main(f1, f2):
    # for different number of executors
    name1 = "pegasos"
    name2 = "mllib"

    b1 = read_result(f1)
    b2 = read_result(f2)

    n = len(b1.keys())
    f, axs = plt.subplots(1, n)

    xlabels = {
        "batchsize": "batchsize(percentage)",
        "size": "size(GB)",
        "executors": "number of executors"
    }

    for i, k in enumerate(b1.keys()):
        ax = axs[i]
        (x1, y1) = b1[k]
        (x2, y2) = b2[k]

        # y1 = 1 / np.array(y1)
        # y2 = 1 / np.array(y2)

        ax.plot(x1, y1, c='b')
        ax.scatter(x1, y1, c='b', marker='o')

        ax.plot(x2, y2, c='r')
        ax.scatter(x2, y2, c='r', marker='x')

        ax.legend([name1, name2], loc=2)
        ax.set_title(k)

        ax.set_xlabel(xlabels[k])
        ax.set_ylabel("time(seconds)")
    # f.legend([name1, name2])
    plt.show()

if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    f1 = os.path.join(path, "grid_pegasos.csv")
    f2 = os.path.join(path, "grid_svm.csv")
    # features = ["batchsize", "executors", "size"]
    main(f1, f2)

    # # features = ["batchsize", "executor", "size", "step"]
    # features = ["step"]
    # plot_loss(path, features)
