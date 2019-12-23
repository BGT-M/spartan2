#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: Shenghua Liu

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from scipy.signal import resample

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

    def toTimeseries(self, freq, attrlabels, numsensors=None, startts=0):
        ''' construct dense matrix for multivariate ts
            # TODO time ticks are also returned from first col of tensorlist
            freq: frequency of the signal
            numsensors: number of signal dimension
            startts: timestamp of the start time
            attrlabels: labels for each dimension
        '''
        print(self.m)
        if numsensors is None:
            tensors = [[] for i in range(self.m)]
        else:
            tensors = [[] for i in range(numsensors)]
        for tensor in self.tensorlist:
            for i in range(len(tensor)):
                tensors[i].append(tensor[i])
        series = np.array(tensors)
        return STTimeseries(freq, series, attrlabels)


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

class STTimeseries:
    def __init__(self, freq, attrlist, attrlabel):
        self.freq = freq
        self.attrlist = attrlist
        self.attrlabel = attrlabel
    
    def show(self, chosen_labels=None):
        if chosen_labels is None:
            sub_dimension = len(self.attrlist)
            actual_dimension = 1
            for label in self.attrlabel:
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = self.attrlabel.index(label)
                plt.title(label)
                plt.plot(self.attrlist[index], label=label)
                plt.legend(loc="best")
                actual_dimension += 1
        else:
            sub_dimension = len(chosen_labels)
            actual_dimension = 1
            for chosen_label in chosen_labels:
                plt.subplot(sub_dimension, 1, actual_dimension)
                for label in chosen_label:
                    index = self.attrlabel.index(label)
                    plt.plot(self.attrlist[index], label=label)
                    plt.legend(loc="best")
                plt.title(chosen_label)
                actual_dimension += 1
        plt.show()

    def resample(self, resampled_freq, show=False, inplace=False):
        origin_list = self.attrlist
        origin_length = len(self.attrlist)
        attr_length = len(self.attrlist[0])
        origin_freq = self.freq
        resampled_list = []
        for index in range(origin_length):
            origin_attr = origin_list[index]
            resampled_attr = resample(origin_attr, int(attr_length/origin_freq*resampled_freq))
            resampled_list.append(resampled_attr)
        resampled_list = np.array(resampled_list)
        if show == True:
            sub_dimension = len(resampled_list)
            actual_dimension = 1
            for label in self.attrlabel:
                x_origin = np.arange(0, attr_length/origin_freq, 1/origin_freq)
                print(len(x_origin))
                x_resampled = np.arange(0, attr_length/origin_freq, 1/resampled_freq)
                print(len(x_resampled))
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = self.attrlabel.index(label)
                plt.title(label)
                plt.plot(x_origin, origin_list[index], 'r-', label='origin')
                plt.plot(x_resampled, resampled_list[index], 'g.', label='resample')
                plt.legend(loc="best")
                actual_dimension += 1
            plt.show()
        if inplace == True:
            self.freq = resampled_freq
            self.attrlist = resampled_list
        else:
            return STTimeseries(resampled_freq, resampled_list, self.attrlabel)
    
    def smooth(self, window_size, show=False, inplace=False):
        pass