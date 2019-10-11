#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#      loader.py
#      Created by @wenchieh  on <11/25/2017>


__author__ = 'wenchieh'


# third-party lib
import numpy as np


class Loader(object):
    def _get_sep_of_file_(self, infn):
        '''
        return the separator of the line.
        :param infn: input file
        '''
        sep = None
        with open(infn, 'r') as fp:
            for line in fp:
                if (line.startswith("%") or line.startswith("#")): continue;
                line = line.strip()
                if (" " in line): sep = " "
                if ("," in line): sep = ","
                if (";" in line): sep = ';'
                if ("\t" in line): sep = "\t"
                break
            fp.close()
        return sep

    def load_edgelist(self, infn_edgelist, comments='#', dtype=int,
                      usecols=(0, 1), idstartzero=True):
        '''
        load edge-list file (node type: int; start index: 0)
        :param infn_edgelist: edgelist file name
        :param comments: comments for the file (at the first line) (default: #)
        :param dtype: all_data type (default: int)
        :param usecols: select specific columns in file (default: [0,1])
        :param idstartzero: whether the id start from zero or not.
        :return: numpy array
        '''
        sep = self._get_sep_of_file_(infn_edgelist)
        data = []

        offset = 0
        if idstartzero != True:
            offset = -1
        with open(infn_edgelist) as fp:
            for line in fp:
                if line.startswith(comments):
                    continue
                arr = line.strip().split(sep)
                elem = []
                for idx in usecols:
                    elem.append(dtype(arr[idx]) + offset)
                data.append(np.array(elem))
            fp.close()

        return np.array(data)

    def load_edgelist2dict(self, infn_edgelist, weighted=False, comments='#',
                           dtype=int, uvindex=(0, 1), idstartzero=True):
        sep = self._get_sep_of_file_(infn_edgelist)
        edges_set = set()
        u2v, v2u = dict(), dict()

        offset = 0
        if idstartzero != True:
            offset = -1
        with open(infn_edgelist) as fp:
            for line in fp:
                if line.startswith(comments):
                    continue
                tokens = line.strip().split(sep)
                u = dtype(tokens[uvindex[0]]) + offset
                v = dtype(tokens[uvindex[1]]) + offset
                if weighted:
                    if u not in u2v:
                        u2v[u] = list()
                    if v not in v2u:
                        v2u[v] = list()
                    u2v[u].append(v)
                    v2u[v].append(u)
                else:
                    if (u, v) not in edges_set:
                        if u not in u2v:
                            u2v[u] = list()
                        if v not in v2u:
                            v2u[v] = list()
                        u2v[u].append(v)
                        v2u[v].append(u)
                        edges_set.add((u, v))

        return u2v, v2u

    def load_adjacency(self, infn_adjlist):
        '''
        load adjacency list [file_adjacencylist] to dict
        :param fn_adjacencylist: input file of adjacency list
        :return: adjacency list dictionary
        '''
        adj_list = dict()

        sep = self._get_sep_of_file_(infn_adjlist)
        with open(infn_adjlist, 'r') as fp:
            for line in fp:
                arr = line.split(sep)
                adj_list[int(arr[0])] = list(map(int, arr[1].split(sep)))
            fp.close()

        return adj_list

    def load_features(self, infn, val_type):
        sep = self._get_sep_of_file_(infn)

        with open(infn, 'r') as fp:
            line = fp.readline().strip()
            m, n = map(int, line[1:].split(sep)[:2])
            fp.close()
        features = np.loadtxt(infn, val_type, delimiter=sep)

        return m, n, features

    def load_degvsfeats(self, infn, val_type, usecols=(0, 1)):
        sep = self._get_sep_of_file_(infn)

        with open(infn, 'r') as fp:
            line = fp.readline()
            splited_line = line[1:].strip().split(sep)
            m, n, mod = int()
            m, n, mod = map(int, line[1:].strip().split(sep))
            usecols = np.unique(usecols)
            degs = []   # np.zeros(m + n, int)
            vals = []   # np.zeros(m + n, val_type)

            for ln in range(m + n):
                arr = fp.readline().strip().split(sep)
                if usecols is None or 0 in usecols:
                    degs.append(int(arr[0]))

                if usecols is None or len(usecols) == mod:
                    rec_val = val_type(arr)
                else:
                    feat_vals = []
                    for k in usecols:
                        if k > 0 and k < mod:
                            feat_vals.append(val_type(arr[k]))
                    if len(feat_vals) == 1:
                        rec_val = feat_vals[0]
                    else:
                        rec_val = feat_vals
                vals.append(rec_val)
            fp.close()

        return m, n, degs, vals

    def load_histogram(self, infn):
        return self.load_multi_histogram(infn)

    def load_multi_histogram(self, infn):
        shape, ticks_vec = [], []
        hist_arr = []

        sep = self._get_sep_of_file_(infn)
        with open(infn, 'r') as ofp:
            shape = list(map(int, ofp.readline().strip()[1:].split(sep)))

            for mod in range(len(shape)):
                ticks = list(map(float, ofp.readline()[1:].strip().split(sep)))
                ticks_vec.append(ticks)

            for line in ofp.readlines():
                toks = line.strip().split(sep)
                hist_arr.append((list(map(int, toks))))
            ofp.close()

        return np.array(shape, int), ticks_vec, np.array(hist_arr, int)

    def load_hcubepos2pts(self, pts2pos_infn, comment='#'):
        pos2pts = dict()
        sep = self._get_sep_of_file_(pts2pos_infn)
        with open(pts2pos_infn, 'r') as ifp:
            index = 0
            for line in ifp:
                if line.startswith(comment):
                    continue
                pos = tuple(map(int, line.strip().split(sep)))
                if pos not in pos2pts:
                    pos2pts[pos] = []
                pos2pts[pos].append(index)
                index += 1
            ifp.close()
        return pos2pts

    def load_pt2pos(self, pts2pos_infn, comment='#'):
        sep = self._get_sep_of_file_(pts2pos_infn)
        ptidx_pos = list()
        with open(pts2pos_infn, 'r') as ifp:
            for line in ifp:
                if line.startswith(comment):
                    continue
                toks = list(map(int, line.strip().split(sep)))
                ptidx_pos.append(toks)
            ifp.close()
        return np.asarray(ptidx_pos)

    def load_describes_parms(sel, describe_infn, describe_cls, mode=2, sep=';'):
        params = list()
        with open(describe_infn, 'r') as ifp:
            for line in ifp:
                line = line.strip()
                splits = line.find(sep)
                desc = describe_cls(mode)
                desc.load(line[splits+1:])
                if desc.is_mix:
                    for k in range(desc.n_components):
                        cov = desc.paras['covs'][k]
                        kparm = list(desc.paras['mus'][k])
                        for m in range(desc.ndim):
                            kparm.extend(cov[m, m:])
                        params.append(kparm)
                else:
                    params.append(desc.compact_parm())
            ifp.close()

        return params