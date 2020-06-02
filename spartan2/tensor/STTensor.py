#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: Shenghua Liu

from .STTimeseries import STTimeseries
from .STGraph import STGraph
from .ioutil import checkfileformat

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

'''Input tensor format:
    format: att1, att2, ..., value1, value2, ...
    comment line started with #
    e.g.
    user obj 1
    ... ...
    if multivariate time series, hasvalue equals to the number of
    time series
    return: tensorlist
'''


class STTensor:
    def __init__(self, tensorlist, hasvalue, value_idx, names):
        self.tensorlist = tensorlist
        self.hasvalue = hasvalue
        self.value_idx = value_idx
        self.names = names
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

    def toTimeseries(self, attrlabels: list = None, numsensors: int = None, freq: int = None, startts: int = None) -> STTimeseries:
        ''' transfer data to time-series type

        Args:
            attrlabels: labels for each dimension
            numsensors: number of signal dimension [except time dimension]
            freq: frequency of the signal, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, freq will not work and will be calculated by the time sequence
            startts: start timestamp, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, startts will not work and will be calculated by the time sequence

        Returns:
            STTimeseries object
        '''
        time = []
        attrlists = np.array(self.tensorlist).T
        if self.names is None and attrlabels is None:
            raise Exception(f'Attrlabels missed.')
        if self.hasvalue == True:
            if self.value_idx is None:
                self.value_idx = 0
            time = attrlists[self.value_idx]
            attrlists = np.delete(attrlists, self.value_idx, axis=0)
            if not self.names is None:
                self.names = list(self.names)
                del self.names[self.value_idx]
        if attrlabels is None:
            attrlabels = self.names
        if numsensors is None:
            tensors = attrlists[:]
        else:
            tensors = attrlists[:numsensors]
        attrlists = np.array(tensors)
        try:
            assert len(attrlabels) == len(tensors)
        except:
            raise Exception(f'Assertions failed with length of labels: {len(attrlabels)} and length of tensors: {len(tensors)}')
        time = np.array(time.astype(np.int64))
        return STTimeseries(time, attrlists, attrlabels, freq=freq, startts=startts)


def loadTensor(name: str, path: str, col_idx: list = None, col_types: list = None,
               hasvalue: bool = 1, value_idx: int = None) -> STTensor:
    '''Load tensor from file.

    Args:
        name: file name
        path: relative or absolute path of directory
        col_idx: id of chosen columns in data file
        col_types: data type of each chosen column
        hasvalue:
            for time series data, this refers to if time column exists
            for graph data, this refers to 
        value_idx: id of value column

    Returns:
        STTensor object
    '''
    if path == None:
        path = "inputData/"
    full_path = os.path.join(path, name)
    if col_types is None:
        if col_idx is None:
            idxtypes = None
        else:
            idxtypes = [(x, str) for x in col_idx]
    else:
        if col_idx is None:
            col_idx = [i for i in range(len(col_types))]
        if len(col_idx) == len(col_types):
            idxtypes = [(x, col_types[x]) for x in col_idx]
        else:
            raise Exception(f"Error: input same size of col_types and col_idx")
    if hasvalue and value_idx is None:
        value_idx = 0
    tensorlist = checkfileformat(full_path, idxtypes)
    names = None
    if type(tensorlist) == tuple:
        tensorlist, names = tensorlist
    printTensorInfo(tensorlist, hasvalue)
    return STTensor(tensorlist, hasvalue, value_idx, names)


def printTensorInfo(tensorlist, hasvalue):
    m = len(tensorlist[0]) - hasvalue
    print(f"Info: Tensor is loaded\n\
           ----------------------\n\
             attr     |\t{m}\n\
             values   |\t{hasvalue}\n\
             nonzeros |\t{len(tensorlist)}\n")
