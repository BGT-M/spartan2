#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#  File: eaglemine_main.py
#  Goal: The main routine of eaglemine algorithm
#      Version:  1.0
#      Goal: Test script
#      Created by @wenchieh  on <12/17/2017>
#

__author__ = 'wenchieh'

# sys
import os
import time
import argparse

# third-party lib
import numpy as np

# project
from .utils.loader import Loader
from .eaglemine_model import EagleMineModel


VERBOSE=True
node2lab = 'nodelabel.out'
hcel2lab = 'hcel2label.out'

def load_hcel_weights(in_hist, in_hcel2avgfeat, mode=2, wtcol_index=1, sep=','):
    loader = Loader()
    _, _, hist_arr = loader.load_histogram(in_hist)

    nhcubes = len(hist_arr)
    hcube2index = dict(zip(map(tuple, hist_arr[:, :2]), range(nhcubes)))
    hcube_weight = np.zeros(nhcubes)  #np.empty((nhcubes, nfeat))
    with open(in_hcel2avgfeat, 'r') as fp:
        for line in fp.readlines():
            if line.startswith('#'):  continue
            tok = line.strip().split(sep)
            pos = tuple(map(int, tok[:mode]))
            hcube_weight[hcube2index[pos]] = float(tok[wtcol_index + mode - 1])
        fp.close()

    return hcube_weight


def eaglemine(in_hist, in_node2hcel, in_hcel2avgfeat, outs,
              strictness=3, desc_voc="dtmnorm", wt_featidx=1, mode=2, mix_comps=2):
    if VERBOSE:
        print("histogram: %s;  hode2pos: %s;  hpos_avgfeat: %s" % (in_hist, in_node2hcel, in_hcel2avgfeat))

    print("EagleMine algorithm")
    print("*****************")
    print("[0]. initialization and loading")
    eaglemodel = EagleMineModel(mode, mix_comps)
    eaglemodel.set_vocabulary(desc_voc)
    eaglemodel.load_histogram(in_hist)

    print("*****************")
    print("[1]. WaterLevelTree")
    start_tm = time.time()
    eaglemodel.leveltree_build(outs, prune_alpha=0.80, verbose=VERBOSE)
    end_tm1 = time.time()
    print("done @ {}".format(end_tm1 - start_tm))

    print("*****************")
    print("[2]. TreeExplore")
    eaglemodel.search(strictness=strictness, verbose=VERBOSE)
    # eaglemodel.save(outfd, '_search_res')
    eaglemodel.post_stitch(verbose=VERBOSE)
    end_tm2 = time.time()
    print("done @ {}".format(end_tm2 - end_tm1))

    if wt_featidx >= 0:
        print("*****************")
        print("[3]. node groups suspicious measure")
        histpos_avgdeg = load_hcel_weights(in_hist, in_hcel2avgfeat, mode, wt_featidx)
        eaglemodel.cluster_weighted_suspicious(histpos_avgdeg, verbose=VERBOSE)

    print("*****************")
    print("saving result")
    eaglemodel.graph_node_cluster(in_node2hcel, os.path.join(outs, node2lab), os.path.join(outs, hcel2lab))
    eaglemodel.cluster_histogram(verbose=VERBOSE)
    eaglemodel.save(outs)

    print('done @ {}'.format(end_tm2 - start_tm))


if __name__ == '__main__':
    path = '../output/'
    in_hist = 'histogram.out'
    in_node2hcel = 'node2hcel.out'
    in_hcel2avgfeat = 'hcel2avgfeat.out'
    desc_voc="dtmnorm"

    wtft = 1   # degree index
    mode, mix_comps = 2, 2
    strictness = 4

    eaglemine(path + in_hist, path + in_node2hcel, path + in_hcel2avgfeat, path,
              strictness, desc_voc, wt_featidx=wtft, mode=mode, mix_comps=mix_comps)
