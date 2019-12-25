#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os
import scipy.sparse.linalg as slin
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class Algorithm():
    def __init__(self, data, alg_obj, model_name):
        self.alg_func = alg_obj
        self.data = data
        self.name = model_name
        self.out_path = "./outputData/"
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

    def showResults(self, plot=False):
        # TODO
        pass


class Holoscope(Algorithm):
    def run(self, k):
        self.alg_func(self.data, self.out_path, self.name, k)


class Fraudar(Algorithm):
    def run(self):
        self.alg_func(self.data, self.out_path, self.name)


class Eaglemine(Algorithm):
    def __init__(self, data, alg_obj, model_name):
        Algorithm.__init__(self, data, alg_obj, model_name)
        self.node_clusters = []

    def run(self, x_feature_array, y_feature_array):
        node_cluster = self.alg_func(x_feature_array, y_feature_array)
        self.node_clusters.append(node_cluster)

    def setbipartite(self, notSquared):
        sm = _get_sparse_matrix(self.data, notSquared)
        hub, s, auth = slin.svds(sm, 1)
        hub = np.squeeze(np.array(hub))
        auth = np.squeeze(np.array(auth))
        self.U = _deal_negative_value(hub)
        self.V = _deal_negative_value(auth)

    def nodes(self, n):
        res = []
        for node_cluster in self.node_clusters:
            if n in node_cluster:
                res.append(node_cluster[n])
        return tuple(res)


class SVDS(Algorithm):
    def run(self, k):
        U, S, V = self.alg_func(self.data, self.out_path, self.name, k)
        return U, S, V


def _get_sparse_matrix(edgelist, notSquared=False):
    edges = edgelist[2]
    edge_num = len(edges)

    # construct the sparse matrix
    xs = [edges[i][0] for i in range(edge_num)]
    ys = [edges[i][1] for i in range(edge_num)]
    data = [1.0] * edge_num

    row_num = max(xs) + 1
    col_num = max(ys) + 1

    if notSquared == False:
        row_num = max(row_num, col_num)
        col_num = row_num

    sm = coo_matrix((data, (xs, ys)), shape=(row_num, col_num), dtype=float)

    return sm


def _deal_negative_value(array):
    if abs(np.max(array)) < abs(np.min(array)):
        array *= -1
    array[array < 0] = 0
    return array
