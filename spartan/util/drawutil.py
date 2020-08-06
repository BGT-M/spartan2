#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   drawutil.py
@Desc    :   Draw functions.
'''

# here put the import lib
import matplotlib.pyplot as plt
import numpy as np

from spartan.tensor import Graph, Timeseries
from spartan import model as _model

# TODO do not import matplotlib function in model file


def plot(model: "Model", *args, **kwargs):
    function_dict = {
        _model.BeatLex: __plot_beatlex
    }
    if function_dict.__contains__(model):
        return function_dict[model](*args, **kwargs)
    else:
        raise Exception(f"Draw functions of model {model} not existed")


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

    nrow, ncol = graph.sm.shape
    if bipartite:
        g = from_biadjacency_matrix(graph.sm)
        # bipartite graph can use other layout, but default is bipartite
        layout = 'bipartite' if layout is None else layout
    else:
        g = nx.from_scipy_sparse_matrix(graph.sm)

    if layout is 'bipartite':
        pos = nx.bipartite_layout(g, nodes=range(nrow))
    else:
        pos = nx_layout[layout](g)

    fig = plt.figure()
    if bipartite:
        nx.draw_networkx(g, pos=pos,
                         node_color=('r'*nrow+'b'*ncol), alpha=0.8)
    else:
        nx.draw_networkx(g, pos=pos, node_color='r', alpha=0.8)

    if labels is not None:
        if isinstance(labels, dict):
            nx.draw_networkx_labels(g, pos=pos, labels=labels)
        else:
            ldict = dict(zip(range(len(labels)), labels))
            nx.draw_networkx_labels(g, pos=pos, labels=ldict)

    return fig


def plot_timeseries(*args, **kwargs):
    import matplotlib.pyplot as plt
    plt.figure()
    __plot_timeseries(plt, *args, **kwargs)
    plt.show()


def __plot_timeseries(plt, series: Timeseries, chosen_labels: list = None):
    if chosen_labels is None:
        sub_dimension = series.dimension
        actual_dimension = 1
        fig, ax = plt.subplots(sub_dimension, 1)
        for index, label in enumerate(series.labels):
            ax[index].set_title(label)
            ax[index].plot(series.time_tensor._data, series.val_tensor._data[index], label=label)
            ax[index].legend(loc="best")
            actual_dimension += 1
    else:
        sub_dimension = len(chosen_labels)
        actual_dimension = 1
        fig, ax = plt.subplots(sub_dimension, 1)
        for chosen_label in chosen_labels:
            for label in chosen_label:
                index = series.labels.index(label)
                ax[actual_dimension-1].plot(series.time_tensor._data, series.val_tensor._data[index], label=label)
            ax[actual_dimension-1].legend(loc="best")
            ax[actual_dimension-1].set_title(', '.join(chosen_label))
            actual_dimension += 1
    plt.xlabel('time/s')


def plot_resampled_series(series: Timeseries, origin_length: int, resampled_length: int, origin_freq: int, resampled_freq: int, origin_list, resampled_list, start):
    plt.figure()
    sub_dimension = len(resampled_list)
    actual_dimension = 1
    for label in series.labels:
        x_origin = np.arange(0, origin_length/origin_freq, 1/origin_freq)
        x_resampled = np.arange(0, resampled_length/resampled_freq, 1/resampled_freq)
        x_origin += start
        x_resampled += start
        plt.subplot(sub_dimension, 1, actual_dimension)
        index = series.labels.index(label)
        plt.title(label)
        plt.plot(x_origin, origin_list[index], 'r-', label='origin')
        plt.plot(x_resampled, resampled_list[index], 'g.', label='resample')
        plt.legend(loc="best")
        actual_dimension += 1
    plt.xlabel('time/s')
    plt.show()


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


def __plot_beatlex(time_series, result):
    import matplotlib.pyplot as plt
    models = result['models']
    __plot_beatlex_vocabulary(plt, models)
    __plot_beatlex_timeseries(plt, time_series, result)
    return plt


def __plot_beatlex_vocabulary(plt, models):
    color = ['r', 'g', 'b', 'c'] * (int(len(models) / 4)+1)
    for i, model in enumerate(models):
        plt.figure()
        plt.plot(model.T, color[i % len(models)])
        plt.title(f"Vocabulary {i}")


def __plot_beatlex_timeseries(plt, series: Timeseries, result: dict):
    plt.figure()
    starts = result['starts']
    ends = result['ends']
    model_num = len(result['models'])
    idx = result['idx']
    color = ['r', 'g', 'b', 'c'] * (int(model_num / 4)+1)
    sub_dimension = series.dimension
    actual_dimension = 1
    _, ax = plt.subplots(sub_dimension, 1)
    for index, label in enumerate(series.labels):
        ax[index].set_title(label)
        if starts is not None and ends is not None:
            for i, _start in enumerate(starts):
                _end = ends[i]
                ax[index].plot(series.time_tensor._data[_start: _end], series.val_tensor._data[index][_start:_end], color[idx[i]], label=label if i == 0 else None)
                ax[index].scatter(series.time_tensor._data[_start], series.val_tensor._data[index][_start], color='black')
        else:
            ax[index].plot(series.time_tensor._data, series.val_tensor._data[index], label=label)
        ax[index].legend(loc="best")
        actual_dimension += 1
    plt.xlabel('time/s')
    return plt
