#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Example tool for extracting graph nodes feature
#  Author: wenchieh
#
#  Project: eaglemine
#      graph.py
#      Version:  1.0
#      Date: December 17 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/17/2017>
#

__author__ = 'wenchieh'

# sys
from abc import ABCMeta, abstractmethod

# third-party lib
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds


class BasicGraph(object):
    __metaclass__ = ABCMeta

    # default setting to control the precision/error of alternative iterative updated SVD
    __ITERS_MAX = 20
    __ITER_PRECISION = 1e-20

    def __init__(self, weighted=False):
        self.weighted = weighted
        self.n_src, self.n_dest = 0, 0
        self.n_edges = 0
        self.edgelist = None
        self.src_hub, self.dest_auth = None, None

    def set_edgelist(self, edgelist_data):
        self.edgelist = edgelist_data
        self.n_edges = len(self.edgelist)
    
    def _get_sparse_matrix(self, matrix_type=lil_matrix):
        m, n = self.get_number_of_node()
        sps_mat = matrix_type((m, n), dtype=float)
        for i, j in self.edgelist:
            if self.weighted:
                if [i, j] not in sps_mat:
                    sps_mat[i, j] = 1.0
                else:
                    sps_mat[i, j] += 1.0
            else:
                sps_mat[i, j] = 1.0
        return sps_mat

    def get_node_degree(self):
        '''
        get the out-degree (in-degree) for source (destination) nodes.
        :return:
        '''
        m, n = self.get_number_of_node()
        self.src_outd, self.dest_ind = np.zeros(m, int), np.zeros(n, int)
        unique_edges = set()

        for i, j in self.edgelist:
            if self.weighted:
                self.src_outd[i] += 1
                self.dest_ind[j] += 1
            else:
                if (i, j) not in unique_edges:
                    self.src_outd[i] += 1
                    self.dest_ind[j] += 1
                    unique_edges.add((i, j))
        return self.src_outd, self.dest_ind

    def _set_hits(self, hub, auth):
        '''
        set the hits score based on given hubness [hub] and authority [auth] containing some post-processing
        :param hub: hubness array (len(hub) == # src_node)
        :param auth: authority aray (len(auth) == # dest_node)
        :return:
        '''
        if abs(np.max(hub)) < abs(np.min(hub)):   # len(np.where(hub <= 0)[0]) >= 0.5 * len(hub):
            hub *= -1
        hub[hub < 0] = 0
        if abs(np.max(auth)) < abs(np.min(auth)): # len(np.where(auth <= 0)[0]) >= 0.5 * len(auth):
            auth *= -1
        auth[auth < 0] = 0

        self.src_hub = hub
        self.dest_auth = auth

    def get_hits_score_alternate(self, iters=15, error=None):
        '''
        calculate HITS score (hub score u1 and authority score v1) for the graph iteratively,
        with [T] iterations or [error] iteration error (precision), one of them should be given at least.
        :param iters: the maximum number of iterations (default=15)
        :param error: the minimum iterate error.
        :return:
        '''
        hub = np.zeros(self.n_src)
        auth = np.ones(self.n_dest) * 1.0 / np.sqrt(self.n_dest)  # with mode = 1

        iters_max = self.__ITERS_MAX
        if (iters is not None) and (iters > 0):
            iters_max = min([self.__ITER_PRECISION, iters])

        error_min = self.__ITER_PRECISION
        if (error is not None) and (error > 0):
            error_min = max([self.__ITER_PRECISION, error])

        # each iteration of HITS algorithm
        k = 0
        while True:
            hub_pre = np.array(hub)
            for e in self.edgelist:
                hub[e[0]] += auth[e[1]]
            hub /= np.sqrt(np.sum(hub ** 2))
            for e in self.edgelist:
                auth[e[1]] += hub[e[0]]
            auth /= np.sqrt(np.sum(auth ** 2))

            iter_error = np.mean(np.abs(hub_pre - hub))
            if (k > iters_max) or (iter_error <= error_min):
                print("iterations: {}, error: {}".format(k, iter_error))
                break
            k += 1

        self._set_hits(hub, auth)
        return self.src_hub, self.dest_auth

    def get_hits_score(self):
        '''
        get HITS score feature for nodes of graph.
        using the SVD algorithm (maybe slowly)
        :return:
        '''
        sps_mat = self._get_sparse_matrix()
        hub, s, auth = svds(sps_mat, k=1, which='LM')
        hub = np.squeeze(np.array(hub))
        auth = np.squeeze(np.array(auth))
        self._set_hits(hub, auth)
        return self.src_hub, self.dest_auth

    @abstractmethod
    def get_number_of_node(self):
        '''
        get the number of the node in graph
        :return:
        '''
        pass

    def get_neighbor_associativity(self):
        '''
        get the neighbors associativity feature (average degree of the 1st-order neighbors) for each nodes of graph.
        :return:
        '''
        self.src_assort = np.zeros(self.n_src, int)
        self.dest_assort = np.zeros(self.n_dest, int)

        for i, j in self.edgelist:
            self.src_assort[i] += self.dest_ind[j]
            self.dest_assort[j] += self.src_outd[i]

        for k in range(self.n_src):
            if self.src_outd[k] > 0:
                self.src_assort[k] = int(self.src_assort[k] / self.src_outd[k])

        for k in range(self.n_dest):
            if self.dest_ind[k] > 0:
                self.dest_assort[k] = int(self.dest_assort[k] / self.dest_ind[k])
        return self.src_assort, self.dest_assort

    @abstractmethod
    def get_corenum(self):
        '''
        get the core number of each node of graph.
        '''
        pass

    def save_features(self, out_fn, src_score, dest_score, sep, header=None):
        m, n = self.n_src, self.n_dest
        with open(out_fn, 'w') as fp:
            fp.writelines('# %d%s%d\n' % (m, sep, n))
            if header is not None:
                fp.writelines('# out/in degree, ' + header + '\n')
            for k in range(m):
                str_line = str(self.src_outd[k]) + sep + str(src_score[k])
                fp.writelines(str_line + '\n')

            for k in range(n):
                str_line = str(self.dest_ind[k]) + sep + str(dest_score[k])
                fp.writelines(str_line + '\n')
            fp.close()

    def save_edgelist(self, out_fn, sep=','):
        with open(out_fn, 'w') as fp:
            for e in self.edgelist:
                str_line = str(e[0]) + sep + str(e[1])
                fp.writelines(str_line + '\n')
            fp.close()

    def save_deg2hits(self, out_fn, sep=','):
        '''
        save HITS feature to file.
        format: ([out-degree][sep][hub], [[in-degree][sep][authority]])
        :param out_fn: output file name
        :param sep: separator of the file (default=',')
        :return:
        '''
        self.save_features(out_fn, self.src_hub, self.dest_auth, sep, "hubness/authority")

    def save_deg2corehum(self, out_fn, sep=','):
        '''
        save the core-number feature to file.
        format: [out(in)-degree][sep][score_number]
        :param out_fn: output file name
        :param sep: separator of the file (default=',')
        :return:
        '''
        self.save_features(out_fn, self.src_cores, self.dest_cores, sep, "src/sink corenum")

    def save_deg2ngbassort(self, out_fn, sep=','):
        '''
        save neighbor associativity feature to file.
        format: [out(in)-degree][sep][neighbor-associativity]
        :param out_fn: output file name
        :param sep: separator of the file (default=',')
        :return:
        '''
        header = "avg in/out degree of 1-st neighbor for src/sink node resp."
        self.save_features(out_fn, self.src_assort, self.dest_assort, sep, )


class BipartiteGraph(BasicGraph):
    def __init__(self, weighted=False):
        super(BasicGraph, self).__init__()
        self.weighted = weighted

    def get_number_of_node(self):
        self.n_src = np.max(self.edgelist[:, 0]) + 1
        self.n_dest = np.max(self.edgelist[:, 1]) + 1
        return self.n_src, self.n_dest

    def get_corenum(self):
        '''
        get the core-number feature of each node.
        construct bipartite with networkx to calculate core-number of each node.
        :return:
        '''
        g = nx.Graph()
        g.add_nodes_from(range(self.n_src), bipartite=0)
        g.add_nodes_from(range(self.n_src, self.n_dest + self.n_src), bipartite=1)
        edges = np.array(self.edgelist)
        edges[:, 1] += self.n_src
        g.add_edges_from(edges)
        cores = nx.core_number(g).values()
        self.src_cores = cores[:self.n_src]
        self.dest_cores = cores[self.n_src:]
        return self.src_cores, self.dest_cores


class UnipartiteGraph(BasicGraph):
    def __init__(self, weighted=False):
        super(BasicGraph, self).__init__()
        self.weighted = weighted

    def set_direct(self, directed=True):
        self.directed = directed

    def set_edgelist(self, edgelist_data):
        self.n_edges = len(edgelist_data)
        if self.directed == False:
            rev_edges = edgelist_data[:, ::-1]
            edgelist_data = np.vstack((edgelist_data, rev_edges))
        self.edgelist = edgelist_data

    def get_node_degree(self):
        sp_mat = self._get_sparse_matrix()
        g = nx.DiGraph()
        xs, ys = sp_mat.nonzero()
        for k in range(len(xs)):
            g.add_edge(xs[k], ys[k], weight=sp_mat[xs[k], ys[k]])
        self.degree = np.asarray(g.degree().values())
        return self.degree

    def get_number_of_node(self):
        max_id = np.max(self.edgelist)
        self.n_src = self.n_dest = max_id + 1
        return self.n_src, self.n_dest

    def get_corenum(self):
        '''
        get the core-number feature of each node.
        construct unipart-graph with networkx to calculate core-number of each node.
        :return:
        '''
        g = nx.Graph()
        g.add_edges_from(self.edgelist)
        self.src_cores = self.dest_cores = nx.core_number(g).values()
        return self.src_cores, self.dest_cores

    def get_triangle(self):
        '''
        get triangle number feature for each node of graph with can be hold only for unipart-graph.
        construct unipart-graph with networkx for calculating the triangles
        :return:
        '''
        g = nx.Graph()
        g.add_edges_from(self.edgelist)
        self.nds_triangles = nx.triangles(g).values()
        return self.nds_triangles

    def get_associate_deg(self):
        '''
        get associate feature of each node of directed unipart-graph (out-degree vs. in-degree.)
        :return:
        '''
        if self.directed == True:
            self.ass_degs = np.zeros((self.n_src, 2), int)
            self.ass_degs[:, 0] = self.src_outd
            self.ass_degs[:, 1] = self.dest_ind
            return self.ass_degs
        else:
            ValueError("No associate features: Only the node of directed "
                       "homogeneous graph have both in-degree and out-degree!")
            return None

    def get_pagerank(self):
        sp_mat = self._get_sparse_matrix()
        g = nx.DiGraph()
        xs, ys = sp_mat.nonzero()
        for k in range(len(xs)):
            g.add_edge(xs[k], ys[k], weight=sp_mat[xs[k], ys[k]])
        self.pagerank = np.asarray(nx.pagerank(g).values())
        return self.pagerank

    def save_deg2pgrk(self, out_fn, sep=','):
        with open(out_fn, 'r') as fp:
            fp.writelines("# #node:{}, #edges:{}, #feature:{}\n".format(self.n_src, self.n_edges, 2))
            fp.writelines("# degree, pagerank\n")
            for k in range(self.n_src):
                line = "{}{}{}".format(self.degree[k], sep, self.pagerank[k])
                fp.writelines(line + '\n')
            fp.close()

    def save_associate_deg(self, out_fn, sep=','):
        with open(out_fn, 'w') as fp:
            fp.writelines("# #node:{}, #edges:{}, #feature:{}\n".format(self.n_src, self.n_edges, 2))
            fp.writelines("# out-degree, in-degree\n")
            for k in range(len(self.ass_degs)):
                line = "{}{}{}".format(self.ass_degs[k, 0], sep ,str(self.ass_degs[k, 1]))
                fp.writelines(line + '\n')
            fp.close()

    def save_deg2triangles(self, out_fn, sep=','):
        '''
        save node triangles-number feature to file.
        format: ([out-degree][sep][triangles], [[in-degree][sep][triangles]])
        :param out_fn: output file name
        :param sep: separator of the file (default=',')
        :return:
        '''
        with open(out_fn, 'w') as fp:
            fp.writelines("# #node:{}, #edges:{}, #feature:{}\n".format(self.n_src, self.n_edges, 2))
            fp.writelines("# degree, #triangles\n")
            for k in range(self.n_src):
                line = "{}{}{}".format(self.degree[k], sep, self.nds_triangles[k])
                fp.writelines(line + '\n')
            fp.close()
