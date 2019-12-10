#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: Shenghua Liu

import os
import sys
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class STTensor:
    def __init__(self, tensorlist, hasvalue):
        self.tensorlist = tensorlist
        self.hasvalue = hasvalue
        'number of columns'
        self.m = len(tensorlist[0])

    def toGraph(self, bipartite=True, weighted=False, rich=False, directed=False, relabel=False):
        '''construct coo sparse matrix of graph from tensorlist
           attributes tuples or matrix are also returned
           bipartite: homogeneous graph or bipartite graph
           weighted: weighted graph or 0-1 adj matrix
           rich: rich graph with edge attributes or not. if yes, requiring
                 tensorlist has more than two attribute columns.
           relabel: relabel ids of graph nodes start from zero
           directed: only effective when bipartite is False, which adj matrix is
                 symmetric
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
            print('Error: list of more than one values is used for graph')
            sys.exit(1)

        if relabel == False:
            row_num = max(xs) + 1
            col_num = max(ys) + 1
            labelmaps = (None, None)
        else:
            print('Error: implement relabel nodes')
            # given labelmaps and inverse maps
            sys.exit(1)

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

    def toTimeseries(self, freq, numsensors=1, startts=0):
        ''' construct dense matrix for multivariate ts
            time ticks are also returned from first col of tensorlist
        '''


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
