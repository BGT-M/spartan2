#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   drawutil.py
@Desc    :   Draw functions.
'''

# here put the import lib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from spartan.tensor import Graph, Timeseries

# TODO do not import matplotlib function in model file


def plot(model: "Model", *args, **kwargs):
    from spartan import model as _model
    function_dict = {
        _model.BeatLex: __plot_beatlex,
        _model.BeatGAN: __plot_beatgan
    }
    if function_dict.__contains__(model):
        function_dict[model](*args, **kwargs)
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
    __plot_timeseries(plt, *args, **kwargs)
    plt.show()


def __plot_timeseries(plt, series: Timeseries, chosen_labels: list = None):
    if chosen_labels is None:
        sub_dimension = series.dimension
        actual_dimension = 1
        fig, ax = plt.subplots(sub_dimension, 1, sharex=True)
        if type(ax) is not np.ndarray:
            ax = [ax]
        for index, label in enumerate(series.labels):
            ax[index].set_title(label)
            ax[index].plot(series.time_tensor._data, series.val_tensor._data[index], label=label)
            ax[index].legend(loc="best")
            actual_dimension += 1
    else:
        sub_dimension = len(chosen_labels)
        actual_dimension = 1
        fig, ax = plt.subplots(sub_dimension, 1, sharex=True)
        if type(ax) is not np.ndarray:
            ax = [ax]
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
    plt = __plot_beatlex_vocabulary(plt, models)
    plt = __plot_beatlex_timeseries(plt, time_series, result)
    plt.show()
    return plt


def __plot_beatlex_vocabulary(plt, models):
    color = ['r', 'g', 'b', 'c'] * (int(len(models) / 4)+1)
    for i, model in enumerate(models):
        plt.figure()
        plt.plot(model.T, color[i % len(models)])
        plt.title(f"Vocabulary {i}")
    return plt


def __plot_beatlex_timeseries(plt, series: Timeseries, result: dict):
    starts = result['starts']
    ends = result['ends']
    model_num = len(result['models'])
    idx = result['idx']
    color = ['r', 'g', 'b', 'c'] * (int(model_num / 4)+1)
    sub_dimension = series.dimension
    actual_dimension = 1
    _, ax = plt.subplots(sub_dimension, 1, sharex=True)
    if sub_dimension == 1:
        ax = [ax]
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


def __plot_beatgan(input, output, heat):
    import matplotlib.pyplot as plt
    sig_in = input.transpose()
    sig_out = output.transpose()
    max_heat = np.max(heat)
    min_heat = np.min(heat)
    x_points = np.arange(sig_in.shape[0])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [6, 1]})
    ax[0].plot(x_points, sig_in, 'k-', linewidth=2.5, label="input signal")
    ax[0].plot(x_points, sig_out, 'k--', linewidth=2.5, label="output signal")
    ax[0].legend(loc="upper right")
    heat = (sig_out-sig_in)**2
    heat_norm = (heat-min_heat)/(max_heat-min_heat)
    heat_norm = np.reshape(heat_norm, (1, -1))
    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()


def histogram_viz(histogram_matrix, x_ticks: list, y_ticks: list, output: str,
                  base: int = 10, x_label: str = None, y_label: str = None):
    '''Plot two-dimensional histogram
    Parameters:
    --------
    :param histogram_matrix: sparse matrix
        histogram data represented with sparse matrix format
    :param x_ticks: list
        ticks of x-axis
    :param y_ticks: list
        ticks of x-axis
    :param base: int
        the logarithmic base for bucketing the graph features.
        Default is 10.
    :param output: str
        output path of the figure
    :param x_label: str
        label of x-axis
        Default is None
    :param y_label: str
        label of y-axis
        Dedault is None
    '''
    import matplotlib.pylab as plt
    from matplotlib.colors import LogNorm

    n, m = histogram_matrix.shape
    fig = plt.figure(figsize=(4.7, 3.8), dpi=96)  # figsize=(8, 6.5),
    plt.pcolormesh(histogram_matrix.toarray(), cmap='jet', norm=LogNorm(), rasterized=True)
    cb = plt.colorbar()
    cb.set_label('counts')
    # for lb in cb.ax.yaxis.get_ticklabels():
    # lb.set_family('Times New roman')
    # lb.set_size(20)

    ax = fig.gca()
    xticks = ax.get_xticks()
    if xticks[-1] > m:
        xticks = xticks[:-1]
    xstep = xticks[1] - xticks[0]
    nw_xtick = []
    for xt in xticks:
        if (xt < m) and (xt % (2*xstep) == 0):
            pws = int(np.log(x_ticks[int(xt)]) / np.log(base))
            if pws != 0:
                nw_xtick.append('%dE%d' % (base, pws))
            else:
                nw_xtick.append('1')
        else:
            nw_xtick.append('')

    nw_ytick = []
    for yt in ax.get_yticks():
        if yt < n:
            yval = y_ticks[int(yt)]
            if yval < 1e4:
                nw_ytick.append(r'%d' % yval)
            else:
                pws = int(np.log10(yval))
                fv = yval * 1.0 / 10**pws
                nw_ytick.append('%.1fE%d' % (fv, pws))

    if nw_xtick[-1] == '':
        nw_xtick[-1] = '%.2f' % x_ticks[-1]
        # nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = '%d' % int(y_ticks[-1])
        # nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick)  # , fontsize=27, family='Times New roman'
    ax.set_yticklabels(nw_ytick)  # , fontsize=27, family='Times New roman'

    if x_label is not None:
        plt.xlabel(x_label, linespacing=12)  # , fontsize=32, family='Times New roman'
    if y_label is not None:
        plt.ylabel(y_label, linespacing=12)  # , fontsize=32, family='Times New roman'

    # fig.set_size_inches(8, 7.3)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output)
    return fig


def __plot_cluster(data, center_pts, data_labels, outliers=list(),
                   core_samples=None, grid=False, outfn=None):
    '''plot clusters of the data'''
    import collections as clct
    import matplotlib.cm as cm
    import matplotlib.pylab as plt

    fig = plt.figure(figsize=(3.8, 3.7), dpi=96)  # figsize=(6.5, 6)
    lab2cnt = clct.Counter(data_labels)
    cmap = cm.get_cmap('Spectral')
    colors = cmap(np.linspace(0, 1, len(lab2cnt)))
    if core_samples is not None:
        core_samples_mask = np.zeros_like(data_labels, dtype=bool)
        core_samples_mask[core_samples] = True
    else:
        core_samples_mask = np.ones_like(data_labels, dtype=bool)

    N_clusters = 0
    keys, values = np.array(list(lab2cnt.keys())), np.array(list(lab2cnt.values()))
    srt_lab = keys[np.argsort(values)][::-1]
    for k, col in zip(srt_lab, colors):
        if k == -1:
            # Black used for noise.
            # col = 'gray'
            continue

        N_clusters += 1
        class_member_mask = (data_labels == k)
        xy = data[class_member_mask & core_samples_mask]
        if len(xy) > 0:
            plt.plot(xy[:, 1], xy[:, 0], 's', color=col, markersize=6)  # markerfacecolor=col, markeredgecolor=col

        mn_xy = np.mean(xy, 0)
        plt.text(mn_xy[1], mn_xy[0], str(k),
                 {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="circle", fc="w", ec="k", pad=0.2, alpha=0.3)})
        xy = data[class_member_mask & (np.invert(core_samples_mask))]
        if len(xy) > 0:
            plt.plot(xy[:, 1], xy[:, 0], 's', color=col, markersize=6)  # markerfacecolor=col, markeredgecolor=col
        if len(center_pts) > 0:
            plt.plot(center_pts[k, 0], center_pts[k, 1], 's', markerfacecolor=col, markeredgecolor='k', markersize=10)

    ubnd, rbnd = np.max(np.vstack((data, outliers)), 0) + 1
    if len(outliers) > 0:
        plt.plot(outliers[:, 1], outliers[:, 0], 'bo', markersize=1)
        uo, ro = np.max(outliers, 0)
        rbnd, ubnd = np.max([rbnd, ro]), np.max([ubnd, uo])

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off', )
    # ax.set_axis_bgcolor('white')   # deprecated method in Matplotlib v2.0
    ax.set_facecolor('white')

    spine_linewidth = 3  # 6 #8#
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    plt.grid(grid)
    plt.xlim((0, rbnd + 0.2))
    plt.ylim((0, ubnd + 0.2))
    plt.tight_layout()
    if outfn is not None:
        plt.savefig(outfn)
        plt.close()
    # plt.show()
    return fig


def clusters_viz(hcel2label: dict, output: str, outlier_label=-1):
    '''Visualize cluster result for histogram
    Parameters:
    -------
    :param hcel2label: dict
        histogram cell to its label
    :param output: str
        output path of the figure
    :param outlier_label: int
        label of the outliers for the 'hcel2label' data.
        Default is -1
    '''
    import warnings
    warnings.filterwarnings("ignore")

    def size_relabel(labels):
        clsdic = {}
        for l in labels:
            if l not in clsdic:
                if l != -1:
                    clsdic[l] = len(clsdic)
                else:
                    clsdic[l] = l
        rlbs = [clsdic[l] for l in labels]
        return np.array(rlbs)

    hcel_lab = np.column_stack((list(hcel2label.keys()), list(hcel2label.values())))
    outs_idx = hcel_lab[:, -1] == outlier_label
    outs = hcel_lab[outs_idx, :-1]
    others, others_lab = hcel_lab[~outs_idx, :-1], hcel_lab[~outs_idx, -1]
    others_lab = size_relabel(others_lab)
    cls_fig = __plot_cluster(others, [], others_lab, outliers=outs[::-1])
    if output is not None:
        cls_fig.savefig(output)
    return cls_fig


def drawHexbin(xs, ys, outfig=None, xscale='log', yscale='log',
               gridsize=200,
               colorscale=True,
               suptitle='Hexagon binning points',
               xlabel='', ylabel=''):
    '''
        xscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the horizontal axis.
    yscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the vertical axis.
        gridsize: [ 100 | integer ]
            The number of hexagons in the x-direction, default is 100. The
            corresponding number of hexagons in the y-direction is chosen such that
            the hexagons are approximately regular. Alternatively, gridsize can be
            a tuple with two elements specifying the number of hexagons in the
            x-direction and the y-direction.
    '''
    xs = np.array(xs) if type(xs) is list else xs
    ys = np.array(ys) if type(ys) is list else ys
    if xscale == 'log' and min(xs) <= 0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
        xg0 = xs > 0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys) <= 0:
        print('[Warning] logscale with nonpositive values in y coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(ys <= 0))))
        yg0 = ys > 0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()
    if colorscale:
        plt.hexbin(xs, ys, bins='log', gridsize=gridsize, xscale=xscale,
                   yscale=yscale, mincnt=1, cmap=plt.cm.jet)
    else:
        plt.hexbin(xs, ys, gridsize=gridsize, xscale=xscale, yscale=yscale,
                   mincnt=1, cmap=plt.cm.jet)

    suptitle = suptitle + ' with a log color scale' if colorscale else suptitle
    plt.title(suptitle)

    cb = plt.colorbar()
    if colorscale:
        cb.set_label('log10(N)')
    else:
        cb.set_label('counts')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if outfig is not None:
        fig.savefig(outfig)
    return fig


def drawRectbin(xs, ys, outfig=None, xscale='log', yscale='log',
                gridsize=200,
                colorscale=True,
                suptitle='Rectangle binning points',
                xlabel='', ylabel=''):
    xs = np.array(xs) if type(xs) is list else xs
    ys = np.array(ys) if type(ys) is list else ys
    if xscale == 'log' and min(xs) <= 0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
        xg0 = xs > 0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys) <= 0:
        print('[Warning] logscale with nonpositive values in y coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(ys <= 0))))
        yg0 = ys > 0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()

    # color scale
    cnorm = matplotlib.colors.LogNorm() if colorscale else matplotlib.colors.Normalize()
    suptitle = suptitle + ' with a log color scale' if colorscale else suptitle

    # axis space
    if isinstance(gridsize, tuple):
        xgridsize = gridsize[0]
        ygridsize = gridsize[1]
    else:
        xgridsize = ygridsize = gridsize
    if xscale == 'log':
        xlogmax = np.ceil(np.log10(max(xs)))
        x_space = np.logspace(0, xlogmax, xgridsize)
    else:
        x_space = xgridsize
    if yscale == 'log':
        ylogmax = np.ceil(np.log10(max(ys)))
        y_space = np.logspace(0, ylogmax, ygridsize)
    else:
        y_space = ygridsize

    hist = plt.hist2d(xs, ys, bins=(x_space, y_space), cmin=1, norm=cnorm,
                      cmap=plt.cm.jet)
    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.title(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    cb = plt.colorbar()
    if colorscale:
        cb.set_label('log10(N)')
    else:
        cb.set_label('counts')

    if outfig is not None:
        fig.savefig(outfig)
    return fig, hist
