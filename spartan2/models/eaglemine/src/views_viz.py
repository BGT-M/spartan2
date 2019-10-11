#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#  File: views_viz.py
#  Goal: provide visualization tools for viewing the clustering result
#
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <12/17/2018>
#

__author__ = 'wenchieh'

# third-party lib
import numpy as np
from scipy.sparse import csr_matrix

# project
from .desc.gaussian_describe import GaussianDescribe
from .desc.dtmnorm_describe import DTMNormDescribe

from .utils.loader import Loader
from .utils.ploter import plot_clusters, plot_heatmap
from .utils.ploter_aux import plot_heatmap_ellipse_covs

VALID_DESCVOCS = ['dtmnorm', 'dmgauss']


def size_relabels(labels):
    clsdic = {}
    for l in labels:
        if l not in clsdic:
            if l != -1: clsdic[l] = len(clsdic)
            else:       clsdic[l] = l
    rlbs = [clsdic[l] for l in labels]
    return np.array(rlbs)

def cluster_view(ins_hcel2lab, outs=None, outlier_lab=-1):
    hcube_label = np.loadtxt(ins_hcel2lab, int, delimiter=',')
    outliers_index = hcube_label[:, -1] == outlier_lab
    outliers = hcube_label[outliers_index, :-1]
    others_cls = hcube_label[~outliers_index]
    labels = others_cls[:, -1]  #size_relabels(others_cls[:, -1])
    cls_fig = plot_clusters(others_cls[:, :-1], [], labels, outliers=outliers[::-1], ticks=False)
    if outs is not None:
        cls_fig.savefig(outs)
    cls_fig.show()

def describe_view(ins_hist, ins_desc, desc_voc, xlab, ylab, outs):
    assert desc_voc in VALID_DESCVOCS
    loader = Loader()
    desc = DTMNormDescribe if desc_voc == 'dtmnorm' else GaussianDescribe
    desc_parms = loader.load_describes_parms(ins_desc, desc, mode=2)
    h_shape, ticks_vec, hist_arr = loader.load_histogram(ins_hist)
    csr_mat = csr_matrix((hist_arr[:, -1], (hist_arr[:, 0], hist_arr[:, 1])), shape=h_shape, dtype=int)
    # plot_heatmap(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=xlabel, ylabel=ylabel, outfn=outfn)
    plot_heatmap_ellipse_covs(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), desc_parms, base=10,
                              scales=(1.5, 3), xlabel=xlab, ylabel=ylab, outfn=outs)


if __name__ == '__main__':
    path = '../output/'
    ins_hist = 'histogram.out'
    ins_desc = 'describe.out'
    ins_hcel2lab = 'hcel2label.out'
    desc_voc = 'dtmnorm'
    xlab, ylab = ["Hubness", "Out-degree"]   # ["Authority", "In-degree"]
    viz_clsv = 'viz_cluster.png'
    viz_desc = 'viz_describes.png'

    cluster_view(path + ins_hcel2lab, path + viz_clsv)
    describe_view(path + ins_hist, path + ins_desc, desc_voc, xlab, ylab, path + viz_desc)
