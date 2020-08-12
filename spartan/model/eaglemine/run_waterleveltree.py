#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

##################################################################
#  WaterLevelTree Test
#  Goal: Test script for WaterLevelTree algorithm
#  Author: wenchieh
#
#  Project: eaglemine
#      waterleveltree.py
#      Version: 
#      Date: November 29 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/30/2017>
#
##################################################################

__author__ = 'wenchieh'


# sys
import os
import time
import argparse

# third-part lib
import numpy as np

# project
from .utils.loader import Loader
from .core.leveltree import LevelTree


VERBOSE = False
tiny_blobs = 'tiny_blob2cnt.out'
contracttree = 'level_tree_contract.out'
prunetree = 'level_tree_prune.out'
refinetree = 'level_tree_refine.out'


def waterleveltree(cel2cnt_arr, outpath):
    tsr_arr = np.asarray(cel2cnt_arr)
    values = np.log2(1.0 + tsr_arr[:, -1])
    max_level = np.max(values)
    step = 0.2

    tree = LevelTree()
    print("Construct raw-tree.")
    tree.build_level_tree(tsr_arr[:, :-1], values, 1.0, max_level, step,
                          verbose=False, outfn=os.path.join(outpath, tiny_blobs))

    print("Refine tree structure.")
    print("a). tree contract")
    tree.tree_contract(VERBOSE)
    tree.save_leveltree(os.path.join(outpath, contracttree))

    print("b). tree pruned")
    tree.tree_prune(alpha=0.8, verbose=VERBOSE)
    tree.save_leveltree(os.path.join(outpath, prunetree))

    print("c). tree node expand")
    tree.tree_node_expand(VERBOSE)
    tree.save_leveltree(os.path.join(outpath, refinetree))

    tree.dump()


if __name__ == '__main__':
    path = '../output/'
    histogram_infn = 'histogram.out'

    loader = Loader()
    print("load data")
    shape, ticks_vec, hist_arr = loader.load_multi_histogram(os.path.join(path, histogram_infn))
    mode = len(shape)
    print("Info: mode:{} shape:{}".format(mode, shape))
    waterleveltree(hist_arr, path)
    print("done!")
