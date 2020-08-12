#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#    graph2histogram.py
#      Version:  1.0
#      Goal: Routine scripts
#      Created by @wenchieh  on <12/27/2018>


__author__ = 'wenchieh'


import numpy as np
from scipy.sparse import csr_matrix

from .utils.loader import Loader
from .tools.histogram_heuristic_generator import HistogramHeuristicGenerator
from .utils.ploter import plot_heatmap, plot_heatmap_graphlab_pgrk, plot_heatmap_2discretes


VERBOSE = True

def histogram_construct(graph_feature, degree_index, outfn_histogram, outfn_pts2pos, outfn_hpos2avgfeat, mode=2):
    n_samples, n_features = graph_feature.shape
    index = np.array([True] * n_samples)
    for mod in range(mode):
        index &= graph_feature[:, mod] > 0
    if VERBOSE:
        print("total shape: {}, valid samples:{}".format(graph_feature.shape, np.sum(index)))

    degree, features = None, None
    graph_feature = graph_feature[index, :]
    if degree_index > 0:
        degree = graph_feature[:,degree_index-1]
        features = np.delete(graph_feature, degree_index-1, axis=1)
        del graph_feature
    else:
        features = graph_feature

    hist_gen = HistogramHeuristicGenerator()
    if degree is not None:
        hist_gen.set_deg_data(degree, features)
        hist_gen.histogram_gen(modeth="degree", N=80, base=10)
    else:
        n_buckets, logarithmic, base = 100, True, 10
        hist_gen.set_data(features)
        hist_gen.histogram_gen(method="N", N=n_buckets, logarithmic=logarithmic, base=base)


    if VERBOSE: hist_gen.dump()
    hist_gen.save_histogram(outfn_histogram)
    hist_gen.save_pts_index(outfn_pts2pos, pts_idx=np.arange(n_samples)[index])
    hist_gen.save_hpos2avgfeat(outfn_hpos2avgfeat)
    print("Graph feature to histogram done!")

def histogram_view(histogram_infn, xlabel, ylabel, outfn=None):
    loader = Loader()
    _shape_, ticks_vec, hist_arr = loader.load_multi_histogram(histogram_infn)
    csr_mat = csr_matrix((hist_arr[:, -1], (hist_arr[:, 0], hist_arr[:, 1])), shape=_shape_, dtype=int)
    plot_heatmap(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=xlabel, ylabel=ylabel, outfn=outfn)
    # plot_heatmap_graphlab_pgrk(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=xlabel, ylabel=ylabel, outfn=outfn)
    # plot_heatmap_2discretes(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=xlabel, ylabel=ylabel, outfn=outfn)
    print('Histogram view done!')


if __name__ == '__main__':
    ins_gfeat = '../example/outd2hub_feature'
    outs = '../output/'
    outs_hist = 'histogram.out'
    ofn_node2hcel = 'node2hcel.out'
    ofn_hcel2avgfeat = 'hcel2avgfeat.out'
    ofn_heatmap = 'heatmap.png'
    x_lab, y_lab = ["Hubness", "Out-degree"]
              # ["Authoritativeness", "In-degree"], ["PageRank", "Degree"], ["Degree", "Triangles"]

    mode = 2
    loader = Loader()
    m, _, gfts = loader.load_features(ins_gfeat, float)
    histogram_construct(gfts[:m], 1, outs + outs_hist, outs + ofn_node2hcel, outs + ofn_hcel2avgfeat, mode)
    histogram_view(outs + outs_hist, x_lab, y_lab, outs + ofn_heatmap)

