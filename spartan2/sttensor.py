#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: Shenghua Liu

from .tensor.STTimeseries import STTimeseries

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200


class STTensor:
    def __init__(self, tensorlist, hasvalue):
        self.tensorlist = tensorlist
        self.hasvalue = hasvalue
        'number of columns'
        self.m = len(tensorlist[0])

    def toGraph(self, bipartite=True, weighted=False, rich=False, directed=False, relabel=False):
        ''' construct coo sparse matrix of graph from tensorlist
            attributes tuples or matrix are also returned
            bipartite: homogeneous graph or bipartite graph
            weighted: weighted graph or 0-1 adj matrix
            rich: rich graph with edge attributes or not. if yes, requiring
                    tensorlist has more than two attribute columns.
            relabel: relabel ids of graph nodes start from zero
            directed: only effective when bipartite is False, which adj matrix is symmetric
        '''
        tl = np.array(self.tensorlist)
        xs = tl[:, 0]
        ys = tl[:, 1]
        edge_num = tl.shape[0]

        if self.hasvalue == 0:
            data = [1] * edge_num
        elif self.hasvalue == 1:
            data = tl[:, -1]
        else:
            raise Exception('Error: list of more than one values is used for graph')

        if relabel == False:
            row_num = max(xs) + 1
            col_num = max(ys) + 1
            labelmaps = (None, None)
        else:
            # given labelmaps and inverse maps
            raise Exception('Error: implement relabel nodes')

        if bipartite == False:
            row_num = max(row_num, col_num)
            col_num = row_num

        dtype = int if weighted == False else float

        sm = coo_matrix((data, (xs, ys)), shape=(row_num, col_num), dtype=dtype)

        if bipartite == False and directed == False:
            'symmetrization sm'
            smt = sm.transpose(copy=True)
            sm = sm.maximum(smt)

        attlist = tensorlist[:, :self.m - hasvalue] if rich is True \
            else None

        return STGraph(sm, weighted, bipartite, rich, attlist, relabel, labelmaps)

    def toTimeseries(self, attrlabels, numsensors=None, freq=None, startts=None):
        ''' transfer data to time-series type
            @params attrlabels: labels for each dimension
            @params numsensors: number of signal dimension [except time dimension]
            @params freq: frequency of the signal, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, freq will not work and will be calculated by the time sequence
            @param startts: start timestamp, default is None
                if time is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, startts will not work and will be calculated by the time sequence

        '''
        time = []
        start = 0
        attrlists = np.array(self.tensorlist).T
        if self.hasvalue == True:
            time = attrlists[0]
            start = 1
        if numsensors is None:
            tensors = attrlists[start:]
        else:
            tensors = attrlists[start:numsensors+start]
        attrlists = np.array(tensors)
        time = np.array(time)
        return STTimeseries(time, attrlists, attrlabels, freq=freq, startts=startts)


class STGraph:
    def __init__(self, sm, weighted, bipartite, rich=False, attlist=None, relabel=False, labelmaps=(None, None)):
        '''
            sm: sparse adj matrix of (weighted) graph
            weighted: graph is weighte or not
            attlist: attribute list with edges, no values
            relabel: relabel or not
            labelmaps: label maps from old to new, and inverse maps from new to
            old
        '''
        self.sm = sm
        self.weighted = weighted
        self.rich = rich
        self.attlist = attlist
        self.relabel = relabel
        self.labelmaps = labelmaps
        self.bipartite = bipartite

    def degrees(self):
        rowdegs, coldegs = self.sm.sum(axis=1), self.sm.sum(axis=0)
        return rowdegs, coldegs.T
