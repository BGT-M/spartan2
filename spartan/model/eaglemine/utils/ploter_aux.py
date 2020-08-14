#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#    ploter_aux.py
#		  auxiliary tools for visualization 
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <11/28/2017>


__author__ = 'wenchieh'

import warnings
warnings.filterwarnings("ignore")

# third-party lib
import numpy as np
from numpy.linalg import eig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm


def _covmat_angle_(cov):
    cov = np.asarray(cov).reshape((2, 2))
    _, v = eig(cov)
    cosine_theta = v[:, 0].dot(np.array([0, 1])) * 1.0 / np.sqrt(np.sum(v[:, 0]**2))
    theta = np.arccos(cosine_theta) * 180 / np.pi
    return theta


def _construct_ellipses_(clusters_para, scale=5.0, SINGLE_PARAS=5):
    N_m = len(clusters_para)

    ell, mus = list(), list()
    left, right, bottom, up = 0, 0, 0, 0
    for k in range(N_m):
        norm = 1.0
        para = clusters_para[k]
        mu = para[:2][::-1]
        cov = np.array([[para[2], para[3]],[para[3], para[4]]])

        angle = _covmat_angle_(cov)
        w, h = scale * np.sqrt(cov[0, 0]) / norm, scale * np.sqrt(cov[1, 1]) / norm
        if w + mu[0] > right:
            right = w + mu[0]
        if h + mu[1] > up:
            up = h + mu[1]
        if -h + mu[1] < bottom:
            bottom = -h + mu[1]

        if mu[1] < 0:    # for  better illustration
            w *= 1.5
            h *= 1.5

        mus.append(mu)
        ell.append(Ellipse(mu, w, h, angle))

    return ell, mus, [left, right, bottom, up]

## ellipses labeled heatmap plot
def plot_heatmap_ellipse(x_vec, y_vec, heatmap, clusters_para, scale=5, base=10,
                         cmap='gray_r', xlabel=None, ylabel=None, outfn=None):
    n, m = heatmap.shape
    fig = plt.figure()
    plt.pcolormesh(heatmap, cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar()
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New roman')
        l.set_size(20)

    ell, mus, bnds = _construct_ellipses_(clusters_para, scale)
    mus = np.asarray(mus)

    ax = fig.gca()
    for k in range(len(ell)):
        e = ell[k]
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(None)
        e.set_fill(False)
        e.set_linewidth(2)
        e.set_color('#ED1C24')
        e.set_linestyle('solid')

    plt.plot(mus[:, 0], mus[:, 1], 'k+', markersize=10, linewidth=5)

    nw_xtick = ['%dE%s' % (base, int(x_vec[int(xt)])) if xt < m else '' for xt in ax.get_xticks()]
    nw_ytick = [r'$%d$' % int(np.power(base, y_vec[int(yt)])) if yt < n else '' for yt in ax.get_yticks()]
    if nw_xtick[-1] == '':
        nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = '%d' % int(y_vec[-1])
        nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick, fontsize=20)
    ax.set_yticklabels(nw_ytick, fontsize=20)

    if xlabel is not None:
        plt.xlabel(xlabel, linespacing=12, fontsize=32, family='Times New roman')
    if ylabel is not None:
        plt.ylabel(ylabel, linespacing=12, fontsize=32, family='Times New roman')

    fig.set_size_inches(8, 7)
    fig.tight_layout()
    if outfn is not None:
        plt.savefig(outfn)
        plt.close()

    return fig

## ellipses (two scale 3 & 5) labeled heatmap plot
def plot_heatmap_ellipse_covs(x_vec, y_vec, heatmap, paras, base=10, scales=(3, 5),
                         cmap='gray_r', xlabel=None, ylabel=None, outfn=None):
    n, m = heatmap.shape
    fig = plt.figure(figsize=(8, 6.5), dpi=96)
    plt.pcolormesh(heatmap, cmap=cmap, norm=LogNorm(), rasterized=True)
    cb = plt.colorbar()
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New roman')
        l.set_size(20)

    ax = fig.gca()
    ell_s1, mus, bnds = _construct_ellipses_(paras, scales[0])
    ell_s2, _, _ = _construct_ellipses_(paras, scales[1])
    ell = list(ell_s1)
    ell.extend(ell_s2)

    for k in range(len(ell)):
        e = ell[k]
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        # e.set_alpha(rnd.rand())
        e.set_facecolor(None)
        e.set_fill(False)
        e.set_linewidth(1.5)
        # e.set_edgecolor(rnd.rand(3))
        # e.set_edgecolor(COLORS[k])
        if k < len(ell_s1):
            # e.set_linewidth(3)
            # e.set_color('#ED1C24')
            e.set_color('#FFF200')
            e.set_linestyle('dashed')
        else:
            # e.set_linewidth(2)
            # e.set_color('#2704FB')
            e.set_color('#F7030C')
            e.set_linestyle('solid')

    xticks = ax.get_xticks()
    if xticks[-1] > m:  xticks = xticks[:-1]

    nw_xtick = ['%dE%d' % (base, int(np.log(x_vec[int(xt)]) / np.log(base)))
                if ((xt < m) and (xt % 20 == 0)) else '' for xt in xticks]
    # nw_ytick = ['%d' % int(y_vec[int(yt)]) if yt < n else '' for yt in ax.get_yticks()]
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
        nw_xtick[-1] = '%.2f' % x_vec[-1]
        # nw_xtick[-1] = '%.2f'%np.power(base, x_vec[-1])
    if nw_ytick[-1] == '':
        nw_ytick[-1] = r'$%d$' % int(y_vec[-1])
        # nw_ytick = '%d' % int(np.power(base, y_vec[-1]))

    ax.set_xticklabels(nw_xtick, fontsize=23, family='Times New roman')   # , fontweight='bold'
    ax.set_yticklabels(nw_ytick, fontsize=23, family='Times New roman')

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=32, family='Times New roman') #, fontweight='bold'
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=32, family='Times New roman')

    # fig.set_size_inches(8, 7)
    fig.tight_layout()
    if outfn is not None:
        plt.savefig(outfn)
        plt.close()

    return fig
