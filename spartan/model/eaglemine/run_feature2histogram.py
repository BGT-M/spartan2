#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#    feature2histogram.py
#      Version:  1.0
#      Goal: Routine scripts
#      Created by @wenchieh  on <12/27/2018>


__author__ = 'wenchieh'

# sys
import argparse

# third-party
import numpy as np
from scipy.sparse import csr_matrix

# project
from .utils.loader import Loader
from .tools.histogram_heuristic_generator import HistogramHeuristicGenerator
from .utils.ploter import plot_heatmap, plot_heatmap_graphlab_pgrk, plot_heatmap_2discretes


VERBOSE = True

def histogram_construct(feats, degidx, outs_hist, outs_node2hcel, outs_hcel2avgfeat, mode=2):
    n_samples, n_features = feats.shape
    index = np.array([True] * n_samples)
    for mod in range(mode):
        index &= feats[:, mod] > 0
    if VERBOSE:
        print("total shape: {}, valid samples:{}".format(feats.shape, np.sum(index)))

    degree, features = None, None
    feats = feats[index, :]
    if degidx > 0:
        degree = feats[:, degidx - 1]
        features = np.delete(feats, degidx - 1, axis=1)
        del feats
    else:
        features = feats

    hist_gen = HistogramHeuristicGenerator()
    if degree is not None:
        hist_gen.set_deg_data(degree, features)
        hist_gen.histogram_gen(method="degree", N=80, base=10)
    else:
        n_buckets = 80
        logarithmic, base = True, 10
        hist_gen.set_data(features)
        hist_gen.histogram_gen(method="N", N=n_buckets, logarithmic=logarithmic, base=base)

    if VERBOSE:
        hist_gen.dump()

    hist_gen.save_histogram(outs_hist)
    hist_gen.save_pts_index(outs_node2hcel, pts_idx=np.arange(n_samples)[index])
    hist_gen.save_hpos2avgfeat(outs_hcel2avgfeat)
    print("done!")

def histogram_view(ins_hist, x_lab, y_lab, outs_viz=None):
    loader = Loader()
    _shape_, ticks_vec, hist_arr = loader.load_multi_histogram(ins_hist)
    csr_mat = csr_matrix((hist_arr[:, -1], (hist_arr[:, 0], hist_arr[:, 1])), shape=_shape_, dtype=int)
    plot_heatmap(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=x_lab, ylabel=y_lab, outfn=outs_viz)
    # plot_heatmap_graphlab_pgrk(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=x_lab, ylabel=y_lab, outfn=outs_viz)
    # plot_heatmap_2discretes(ticks_vec[1], ticks_vec[0], csr_mat.toarray(), xlabel=x_lab, ylabel=y_lab, outfn=outs_viz)
    print('done!')


def demo():
    mode = 2
    outs = '../output/'
    ins_gfeat = '../example/outd2hub_feature'
    outs = '../output/'
    ofn_hist = 'histogram.out'
    ofn_node2hcel = 'node2hcel.out'
    ofn_hcel2avgfeat = 'hcel2avgfeat.out'
    ofn_heatmap = 'heatmap.png'
    x_lab, y_lab = ["Hubness", "Out-degree"] # ["Authority", "In-degree"], ["PageRank", "Degree"], ["Degree", "Triangles"]

    loader = Loader()
    m, _, gfts = loader.load_features(ins_gfeat, float)
    histogram_construct(gfts[:m], 1, outs + ofn_hist, outs + ofn_node2hcel, outs + ofn_hcel2avgfeat, mode)
    histogram_view(outs + ofn_hist, x_lab, y_lab, outs + ofn_heatmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct histogram for given TWO-dimension feature",
                                     usage="python feature2histogram.py [ins], [degidx] [x_idx] [y_idx] [x_lab] [y_lab]"
                                           "[outs_hist] [outs_node2hcel] [outs_hcel2avgfeat] [outs_viz] [delimiter] [comments]")
    parser.add_argument("ins", type=str, help="input feature path")
    parser.add_argument("degidx", type=int, default=0, help="feature index if contain (in/out-) degree for graph else 0")
    parser.add_argument("x_idx", type=int, help="feature index for x axis")
    parser.add_argument("y_idx", type=int, help="feature index for y axis")
    parser.add_argument("x_lab", type=str, help="feature label for x axis")
    parser.add_argument("y_lab", type=str, help="feature label for y axis")
    parser.add_argument("outs_hist", type=str, help="output path for result of feature-to-histogram ")
    parser.add_argument("outs_node2hcel", type=str, help="output path for result of point to cell of histogram")
    parser.add_argument("outs_hcel2avgfeat", type=str, help="output path for result of average feature of each cell in histogram")
    parser.add_argument("outs_viz", type=str, help="output path for result of histogram view")
    parser.add_argument("delimiter", type=str, default=',', help="delimiter of the input")
    parser.add_argument("comments", type=str, default='%', help="comments character of the input")
    args = parser.parse_args()

    mode = 2
    feat = np.loadtxt(args.ins, float, args.comments, args.delimiter, usecols=[args.x_idx, args.y_idx])
    histogram_construct(feat, args.degidx, args.outs_hist, args.outs_node2hcel, args.outs_hcel2avgfeat, mode)
    histogram_view(args.outs_hist, args.x_lab, args.y_lab, args.outs_viz)
