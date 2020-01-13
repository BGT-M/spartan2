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


class STTimeseries:
    def __init__(self, time, attrlists, attrlabels, freq=None, startts=None):
        ''' init STTimeseries object
            @param time: time dimension of data
            @param attrlists: signal dimensions of data
            @param attrlabels: labels of data, positions corresponding to data
            @params freq: frequency of the signal, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, freq will not work and will be calculated by the time sequence
            @param startts: start timestamp, default is None
                if time is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, startts will not work and will be calculated by the time sequence
        '''
        self.length = len(attrlists[0])
        if len(time) == 0:
            if freq is None:
                raise Exception('Parameter freq not provided')
            if startts is None:
                raise Exception('Parameter startts not provided')
            self.freq = freq
            self.startts = startts
            self.timelist = np.arange(startts, 1/freq*self.length, 1 / freq)
        else:
            self.timelist = time
            self.freq = int(len(time) / (time[-1] - time[0]))
            self.startts = time[0]
        self.attrlists = attrlists
        self.attrlabels = attrlabels

    def show(self, chosen_labels=None):
        ''' draw series data with matplotlib.pyplot
            @type chosen_labels: [[]]
            @param chosen_labels:
                if None, draw all the attrs in subgraph;
                or treat all 1-dimen array as subgraphs and entries in each array as lines in each subgraph
        '''
        if chosen_labels is None:
            sub_dimension = len(self.attrlists)
            actual_dimension = 1
            for label in self.attrlabels:
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = self.attrlabels.index(label)
                plt.title(label)
                plt.plot(self.timelist, self.attrlists[index], label=label)
                plt.legend(loc="best")
                actual_dimension += 1
        else:
            sub_dimension = len(chosen_labels)
            actual_dimension = 1
            for chosen_label in chosen_labels:
                plt.subplot(sub_dimension, 1, actual_dimension)
                for label in chosen_label:
                    index = self.attrlabels.index(label)
                    plt.plot(self.timelist, self.attrlists[index], label=label)
                    plt.legend(loc="best")
                plt.title(chosen_label)
                actual_dimension += 1
        plt.show()

    def resample(self, resampled_freq, show=False, inplace=False):
        ''' resample series data with a new frequency, acomplish on the basis of scipy.signal.sample
            @param resampled_freq: resampled frequency
            @param show: if True, show the resampled signal with matplotlib.pyplot
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
        '''
        origin_list = self.attrlists
        origin_length = len(self.attrlists)
        attr_length = self.length
        origin_freq = self.freq
        resampled_list = []
        for index in range(origin_length):
            origin_attr = origin_list[index]
            resampled_attr = resample(origin_attr, int(attr_length/origin_freq*resampled_freq))
            resampled_list.append(resampled_attr)
        resampled_list = np.array(resampled_list)
        resampled_length = len(resampled_list[0])
        resampled_time = np.arange(self.startts, self.startts + 1 / resampled_freq * resampled_length, 1 / resampled_freq)
        if show == True:
            sub_dimension = len(resampled_list)
            actual_dimension = 1
            for label in self.attrlabels:
                x_origin = np.arange(0, attr_length/origin_freq, 1/origin_freq)
                x_resampled = np.arange(0, attr_length/origin_freq, 1/resampled_freq)
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = self.attrlabels.index(label)
                plt.title(label)
                plt.plot(x_origin, origin_list[index], 'r-', label='origin')
                plt.plot(x_resampled, resampled_list[index], 'g.', label='resample')
                plt.legend(loc="best")
                actual_dimension += 1
            plt.show()
        if inplace == True:
            self.time = resampled_time
            self.attrlists = resampled_list
            self.freq = resampled_freq
            self.length = resampled_length
        else:
            return STTimeseries(resampled_time, resampled_list, self.attrlabels.copy(), resampled_freq, self.startts)

    def combine(self, combined_series, inplace=True):
        ''' combine series data which have the same frequency
            @param combined_series: series to be combined
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
        '''
        if not self.freq == combined_series.freq:
            raise Exception(f'Frequency not matched, with {self.freq} and {combined_series.freq}')
        if inplace:
            for label in combined_series.attrlabels:
                if label in self.attrlabels:
                    for i in range(1, 10000):
                        if not (label + '_' + str(i)) in self.attrlabels:
                            self.attrlabels.extend([label + '_' + str(i)])
                            break
                else:
                    self.attrlabels.extend([label])
            self.attrlists = np.concatenate([self.attrlists, combined_series.attrlists])
        else:
            attrlabels = self.attrlabels.copy()
            for label in combined_series.attrlabels:
                if label in attrlabels:
                    for i in range(1, 10000):
                        if not (label + '_' + str(i)) in attrlabels:
                            attrlabels.extend([label + '_' + str(i)])
                            break
                else:
                    attrlabels.extend([label])
            attrlists = np.concatenate([self.attrlists, combined_series.attrlists])
            return STTimeseries(self.timelist.copy(), attrlists, attrlabels, self.freq, self.startts)

    def cut(self, attrs=None, start=None, end=None, form='point', inplace=False):
        ''' cut columns in time dimension
            @type attrs: array
            @param attrs: default is None, columns to be cut
                if not None, attr sprcified in attrs will be cut AND param inplace will be invalid
            @param start: default is None, start position
                if start is None, cut from the very front position
            @param end: default is None, end position
                if end is None, cut to the very last position
            @param form: default is point, type of start and end
                if "point", start and end would mean absolute positions of columns
                if "time", start and end would mean timestamp and need to multiply frequenct to get the absolute positions
            @param inplace: default if False, IF attrs is not None, this param will be invalid
                if False, function will return a new STTimeseiries object
                if True, function will make changes in current STTimeseries object
        '''
        if attrs is None:
            templabels = self.attrlabels
            templists = self.attrlists
        else:
            templabels = []
            templists = []
            for attr in attrs:
                if not attr in self.attrlabels:
                    raise Exception(f'Attr {attr} is not found')
                templabels.append(attr)
                index = self.attrlabels.index(attr)
                templists.append(self.attrlists[index])
        if form == 'point':
            start = start
            end = end
        elif form == 'time':
            if not start is None:
                start = start * self.freq
            if not end is None:
                end = end * self.freq
        else:
            raise Exception('Value of parameter form is not defined!')
        if start is None:
            start = 0
        if end is None:
            end = self.length
        timelist = self.timelist.copy()
        timelist = timelist[start:end]
        templists = [attr[start:end] for attr in templists]
        templists = np.array(templists)
        if attrs is None and inplace:
            self.attrlists = templists
            self.timelist = timelist
            self.startts = timelist[0]
            self.length = len(templists[0])
        else:
            return STTimeseries(timelist, templists, templabels)



    def smooth(self, window_size, show=False, inplace=False):
        pass

    def savefile(self, name, path=None, attrs=None, annotation=None):
        ''' save current time series object as a tensor file, time column [if exists] shall always be stored as the first column
            @param name: name of the file to be saved
            @param path: default is None, parent directory
            @param attr: default is None
                if assigned, only save required columns
            @param annotation: annotations which will be saved at the first line of the file
        '''
        if path is None:
            path = f'./{name}.tensor'
        else:
            path = f'{path}{name}.tensor'
        if attrs is None:
            templists = self.attrlists
            templabels = self.attrlabels
        else:
            templists = []
            templabels = attrs
            for attr in attrs:
                if not attr in self.attrlabels:
                    raise Exception(f'Attr {attr} not found!')
                index = self.attrlabels.index(attr)
                templists.append(self.attrlists[index])
            templists = np.array(templists)
        templists = templists.T
        templists = [','.join(map(lambda x:str(x), t)) for t in templists]
        with open(path, 'w') as writer:
            if not annotation is None:
                writer.write(f'# {annotation} \n')
            writer.write(f'# time {" ".join(templabels)} \n')
            for i in range(self.length):
                writer.write(f'{self.timelist[i]},{templists[i]}\n')
