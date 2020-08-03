#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   drawutil.py
@Desc    :   Draw functions.
'''

# here put the import lib
import matplotlib.pyplot as plt
import numpy as np

from spartan.tensor import Graph

# TODO do not import matplotlib function in model file


def plot_graph(graph: Graph, layout=None, bipartite=False, labels=None,
               *args, **kwargs):
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
    nx_layout = {
        None: nx.random_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'spectral': nx.spectral_layout,
        'spring': nx.spring_layout,
        'bipartite': nx.bipartite_layout
    }
    if bipartite:
        g = from_biadjacency_matrix(graph.sm)
    else:
        g = nx.from_scipy_sparse_matrix(graph.sm)
    pos = nx_layout[layout](g)
    nx.draw_networkx(g, pos=pos)
    if labels is not None:
        nx.draw_networkx_labels(g, pos=pos, labels=labels)


def drawEigenPulse(densities: list = [], figpath: str = None):
    xs = range(len(densities))
    plt.plot(xs, densities, label='density')
    plt.xlabel('window idx')
    plt.ylabel('density')

    thres = np.mean(densities) + 3 * np.std(densities)
    plt.hlines(thres, min(xs), max(xs), linestyles='dashed',
               colors='yellow', label='threshold')
    plt.legend()
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath)
