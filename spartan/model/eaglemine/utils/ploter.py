#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#      ploter.py
#		  visualization tools
#      Created by @wenchieh  on <11/25/2017>


__author__ = 'wenchieh'

# sys
import warnings
warnings.filterwarnings("ignore")
import collections as clct

# third-party lib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm


# Plot item degree distribution (frequency / P(x>=d)) vs. degree
def plot_degree_dist(degree, value, xlabel, ylabel, color_marker='b.',
                     title=None, outfn=None):
    fig = plt.figure()
    plt.loglog(degree, value, color_marker, markersize=2, markeredgecolor=None)
    plt.xlabel(xlabel, linespacing=12, fontsize=18)
    plt.ylabel(ylabel, linespacing=12, fontsize=18)
    plt.tight_layout()
    if title is not None: plt.title(title, fontsize=18, y=1.05)
    if outfn is not None:
        fig.savefig(outfn)
        plt.close()
    return fig


def plot_log2pdf(feature, xlabel, color_marker='b.', ylabel='Frequency', outfn=None):
    ft2cnt = dict()
    for f in feature:
        ft2cnt[f] = ft2cnt.get(f, 0) + 1

    fig = plt.figure()
    plt.loglog(ft2cnt.keys(), ft2cnt.values(), color_marker)
    plt.xlabel(xlabel, linespacing=12, fontsize=18)
    plt.ylabel(ylabel, linespacing=12, fontsize=18)
    plt.tight_layout()
    if outfn is not None:
        plt.savefig(outfn)
        plt.close()
    # plt.show()
    return fig


def plot_clusters(data, center_pts, data_labels, outliers=list([]),
                  core_samples=None, grid=False, ticks=True, outfn=None):
    fig = plt.figure(figsize=(6.5, 6), dpi=96)
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
            plt.plot(xy[:, 1], xy[:, 0], 's', color=col, markersize=6) #markerfacecolor=col, markeredgecolor=col

        mn_xy = np.mean(xy, 0)
        plt.text(mn_xy[1], mn_xy[0], str(k),
                 {'color': 'k', 'fontsize': 18, 'ha': 'center', 'va': 'center',
                  'bbox': dict(boxstyle="circle", fc="w", ec="k", pad=0.2, alpha=0.3)} )
        xy = data[class_member_mask & (np.invert(core_samples_mask))]
        if len(xy) > 0:
            plt.plot(xy[:, 1], xy[:, 0], 's', color=col, markersize=6) #markerfacecolor=col, markeredgecolor=col
        if len(center_pts) > 0:
            plt.plot(center_pts[k, 0], center_pts[k, 1], 's',
                     markerfacecolor=col, markeredgecolor='k', markersize=10)

    ubnd, rbnd = np.max(np.vstack((data, outliers)), 0) + 1
    if len(outliers) > 0:
        plt.plot(outliers[:, 1], outliers[:, 0], 'bo', markersize=1)
        uo, ro = np.max(outliers, 0)
        rbnd, ubnd = np.max([rbnd, ro]), np.max([ubnd, uo])

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', bottom='off', left='off',
                   labelbottom='off', labelleft='off', )
    # ax.set_axis_bgcolor('white')   # deprecated method in Matplotlib v2.0
    ax.set_facecolor('white')

    spine_linewidth = 6 #8#
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


def plot_hexbin_heatmap(xs, ys, xlabel, ylabel, xscale='log',
                        yscale='log', gridsize=100, colorscale=True, outfn=None):
    '''
    xscale: [ ‘linear’ | ‘log’ ]
        Use a linear or log10 scale on the horizontal axis.
    yscale: [ ‘linear’ | ‘log’ ]
        Use a linear or log10 scale on the vertical axis.
    gridsize: [ 100 | integer ]
        The number of hexagons in the x-direction, default is 100. The
        corresponding number of hexagons in the y-direction is chosen such that
        the hexagons are approximately regular. Alternatively, gridsize can be
        a tuple with two elements specifying the number of hexagons in the
        x-direction and the y-direction.
    '''
    fig = plt.figure()

    if colorscale:
        plt.hexbin(xs, ys, bins='log', gridsize=gridsize, xscale=xscale,
                   yscale=yscale, mincnt=1, cmap=cm.get_cmap('jet'))
        cb = plt.colorbar()
        cb.set_label(r'$log_{10}(N)$', fontsize=16)
    else:
        plt.hexbin(xs, ys, gridsize=gridsize, xscale=xscale, yscale=yscale,
                   mincnt=1, cmap=cm.get_cmap('jet'))
        cb = plt.colorbar()
        cb.set_label('counts')
    plt.xlabel(xlabel, linespacing=12, fontsize=18)
    plt.ylabel(ylabel, linespacing=12, fontsize=18)
    plt.tight_layout()

    if outfn is not None:
        fig.savefig(outfn)
        plt.close()
    # plt.show()
    return fig


def plot_heatmap(x_vec, y_vec, heatmap, base=10, xlabel=None, ylabel=None, outfn=None):
    n, m = heatmap.shape
    fig = plt.figure(figsize=(8, 6.5), dpi=96)
    plt.pcolormesh(heatmap, cmap='jet', norm=LogNorm(), rasterized=True)
    cb = plt.colorbar()
    for lb in cb.ax.yaxis.get_ticklabels():
        lb.set_family('Times New roman')
        lb.set_size(20)

    ax = fig.gca()
    xticks = ax.get_xticks()
    if xticks[-1] > m:  xticks = xticks[:-1]
    xstep = xticks[1] - xticks[0]
    nw_xtick = []
    for xt in xticks:
        if (xt < m) and (xt % (2*xstep) == 0):
            pws = int(np.log(x_vec[int(xt)]) / np.log(base))
            if pws != 0:
                nw_xtick.append('%dE%d' % (base, pws))
            else:
                nw_xtick.append('1')
        else:
            nw_xtick.append('')

    nw_ytick = []
    for yt in ax.get_yticks():
        if yt < n:
            yval = y_vec[int(yt)]
            if yval < 1e4:
                nw_ytick.append(r'%d' % yval)
            else:
                pws = int(np.log10(yval))
                fv = yval * 1.0 / 10**pws
                nw_ytick.append('%.1fE%d'%(fv, pws))

    if nw_xtick[-1] == '':
        nw_xtick[-1] = '%.2f'%x_vec[-1]
        # nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = '%d' % int(y_vec[-1])
        # nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick, fontsize=27, family='Times New roman')
    ax.set_yticklabels(nw_ytick, fontsize=27, family='Times New roman')

    if xlabel is not None:
        plt.xlabel(xlabel, linespacing=12, fontsize=32, family='Times New roman')
    if ylabel is not None:
        plt.ylabel(ylabel, linespacing=12, fontsize=32, family='Times New roman')

    # fig.set_size_inches(8, 7.3)
    fig.tight_layout()
    if outfn is not None:
        fig.savefig(outfn)
    return fig


# plot the ROC/AUC curves given #N True-Positive-Rate & False-Positive-Rate Series
#     colors, markers for each serie
def plot_roc_curve(NTPRS, NFPRS, legends_str, colors, markers, outfn=None):
    # colors=['r', 'g', 'm', 'b', 'c', 'k'],
    # markers=[None, None, None, None, None, None, '>', 'v', '+']
    N_med = len(NTPRS)
    fig = plt.figure()
    for k in range(N_med):
        plt.plot(NFPRS[k], NTPRS[k], color=colors[k], linewidth=2,
                 linestyle='-', marker=markers[k], label=legends_str[k])
    # plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.plot([0, 1], [0, 1], color='k', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('False positive rate', linespacing=12, fontsize=32, family='Times New roman', fontweight='bold')
    plt.ylabel('True positive rate', linespacing=12, fontsize=32, family='Times New roman', fontweight='bold')
    plt.legend(loc='lower right', ncol=1, fancybox=False, shadow=False, fontsize=22)
    plt.tight_layout()
    if outfn is not None:
        plt.savefig(outfn)
        plt.close()
    # plt.show()
    return fig


def plot_heatmap_graphlab_pgrk(x_vec, y_vec, heatmap, base=10, xlabel=None, ylabel=None, outfn=None):
    n, m = heatmap.shape
    fig = plt.figure(figsize=(8, 6.5), dpi=96)
    plt.pcolormesh(heatmap, cmap='jet', norm=LogNorm(), rasterized=True)

    cb = plt.colorbar()
    for lb in cb.ax.yaxis.get_ticklabels():
        lb.set_family('Times New roman')
        lb.set_size(20)

    ax = fig.gca()
    xticks = ax.get_xticks()
    if xticks[-1] > m:  xticks = xticks[:-1]
    ### for graph-lab  pagerank featrue
    nw_xtick = []
    for xt in xticks: #ax.get_xticks():
        if ((xt < m) and (xt % 40 == 0)):
            xval = x_vec[int(xt)]
            if xval < 1:
                nw_xtick.append(r'%.2f' % xval)
            elif xval < 100:
                nw_xtick.append(r'%.1f' % xval)
            else:
                pws = int(np.log10(xval))
                fv = xval * 1.0 / 10**pws
                nw_xtick.append(r'%.1fE%d'%(fv, pws))
        else:
            nw_xtick.append('')

    nw_ytick = []
    for yt in ax.get_yticks():
        if yt < n:
            yval = y_vec[int(yt)]
            if yval < 1e3:
                nw_ytick.append(r'%d' % yval)
            else:
                pws = int(np.log10(yval))
                fv = yval * 1.0 / 10**pws
                nw_ytick.append('%.1fE%d'%(fv, pws))

    if nw_xtick[-1] == '':
        nw_xtick[-1] = '%.2f'%x_vec[-1]
        # nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = '%d' % int(y_vec[-1])
        # nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick, fontsize=23, family='Times New roman')
    ax.set_yticklabels(nw_ytick, fontsize=23, family='Times New roman')

    if xlabel is not None:
        plt.xlabel(xlabel, linespacing=12, fontsize=32, family='Times New roman')
    if ylabel is not None:
        plt.ylabel(ylabel, linespacing=12, fontsize=32, family='Times New roman')

    # fig.set_size_inches(8, 7.3)
    fig.tight_layout()
    if outfn is not None:
        fig.savefig(outfn)
        plt.close()
    return fig


def plot_heatmap_2discretes(x_vec, y_vec, heatmap, base=10, xlabel=None, ylabel=None, outfn=None):
    n, m = heatmap.shape
    fig = plt.figure(figsize=(8, 6.5), dpi=96)
    plt.pcolormesh(heatmap, cmap='jet', norm=LogNorm(), rasterized=True)
    cb = plt.colorbar()

    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New roman')
        l.set_size(20)

    ax = fig.gca()
    xticks = ax.get_xticks()
    if xticks[-1] > m:  xticks = xticks[:-1]

    ### for triangle_count / degree  feature
    nw_xtick = []
    for xt in xticks:
        if ((xt < m) and (xt % 20 == 0)):
            xval = x_vec[int(xt)]
            if xval < 1e3:
                nw_xtick.append(r'%d' % xval)
            else:
                pws = int(np.log10(xval))
                fv = xval * 1.0 / 10**pws
                nw_xtick.append(r'%.1fE%d'%(fv, pws))
        else:
            nw_xtick.append('')

    nw_ytick = []
    for yt in ax.get_yticks():
        if yt < n:
            yval = y_vec[int(yt)]
            if yval < 1e3:
                nw_ytick.append(r'%d' % yval)
            else:
                pws = int(np.log10(yval))
                fv = yval * 1.0 / 10**pws
                nw_ytick.append('%.1fE%d'%(fv, pws))

    if nw_xtick[-1] == '':
        nw_xtick[-1] = '%.2f'%x_vec[-1]
        # nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = '%d' % int(y_vec[-1])
        # nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick, fontsize=27, family='Times New roman')
    ax.set_yticklabels(nw_ytick, fontsize=27, family='Times New roman')

    if xlabel is not None:
        plt.xlabel(xlabel, linespacing=12, fontsize=32, family='Times New roman')
    if ylabel is not None:
        plt.ylabel(ylabel, linespacing=12, fontsize=32, family='Times New roman')

    # fig.set_size_inches(8, 7.3)
    fig.tight_layout()
    if outfn is not None:
        fig.savefig(outfn)
        plt.close()
    return fig
