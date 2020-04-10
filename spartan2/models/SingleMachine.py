#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os

from .holoscope.holoscopeFraudDect import Ptype, HoloScope, scoreLevelObjects
from .fraudar.greedy import logWeightedAveDegree, np
from .ioutil import saveSimpleListData, loadedgelist2sm
from .eaglemine.src.eaglemine_main import eaglemine
from .eaglemine.src.graph2histogram import histogram_construct
from .eaglemine.src.views_viz import cluster_view
from .beatlex.Beatlex import BeatLex
import scipy.sparse.linalg as slin


class AnomalyDetection:
    @staticmethod
    def HOLOSCOPE(graph, out_path, file_name, k):
        sparse_matrix = graph.sm
        sparse_matrix = sparse_matrix.asfptype()
        ptype = [Ptype.freq]
        alg = 'fastgreedy'
        qfun, b = 'exp', 32  # 10 #4 #8 # 32
        tunit = 'd'
        bdres = HoloScope(sparse_matrix, alg, ptype, qfun=qfun, b=b, tunit=tunit, nblock=k)
        opt = bdres[-1]
        for nb in range(k):
            res = opt.nbests[nb]
            print('block{}: \n\tobjective value {}'.format(nb + 1, res[0]))
            export_file = out_path + file_name + '.blk{}'.format(nb + 1)
            saveSimpleListData(res[1][0], export_file + '.rows')
            saveSimpleListData(res[1][1], export_file + '.colscores')
            levelcols = scoreLevelObjects(res[1][1])
            saveSimpleListData(levelcols, export_file + '.levelcols')

    @staticmethod
    def FRAUDAR(graph, out_path, file_name):
        sparse_matrix = graph.sm
        sparse_matrix = sparse_matrix.asfptype()
        res = logWeightedAveDegree(sparse_matrix)
        print(res)
        np.savetxt("%s.rows" % (out_path + file_name, ), np.array(list(res[0][0])), fmt='%d')
        np.savetxt("%s.cols" % (out_path + file_name, ), np.array(list(res[0][1])), fmt='%d')
        print("score obtained is ", res[1])

    @staticmethod
    def EAGLEMINE(x_feature_array, y_feature_array):
        tempdir = "temp/"
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)
        tempfile = {
            "feature": "outd2hub_feature",
            "histogram": "histogram.out",
            "node2hcel": "node2hcel.out",
            "hcel2avgfeat": "hcel2avgfeat.out"
        }

        outdir = "outputData/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        output = {
            "hcel2label": "hcel2label.out",
            "viz_clsv": "viz_cluster.png",
            "node2lab": "nodelabel.out"
        }
        array_length = len(x_feature_array)
        mix_array = [[y_feature_array[i], x_feature_array[i]] for i in range(array_length)]
        graph_ndft = np.array(mix_array, dtype=np.float64)
        histogram_construct(graph_ndft, int(1), tempdir + tempfile["histogram"], tempdir + tempfile["node2hcel"], tempdir + tempfile["hcel2avgfeat"])
        eaglemine(tempdir + tempfile["histogram"], tempdir + tempfile["node2hcel"], tempdir + tempfile["hcel2avgfeat"], outdir, int(4))
        cluster_view(outdir + output["hcel2label"], outdir + output["viz_clsv"])
        node_cluster = {}
        with open(outdir + output["node2lab"], 'r') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("#"):
                    continue
                else:
                    coords = line.split(',')
                    node_id = int(coords[0])
                    cluster_id = int(coords[1])
                    if cluster_id not in node_cluster:
                        node_cluster[cluster_id] = []
                    node_cluster[cluster_id].append(node_id)

        return node_cluster


class Decomposition:
    @staticmethod
    def SVDS(mat, out_path, file_name, k):
        sparse_matrix = mat
        sparse_matrix = sparse_matrix.asfptype()
        RU, RS, RVt = slin.svds(sparse_matrix, k)
        export_file = out_path + file_name
        #saveSimpleListData(res[0], export_file + '.leftSV')
        #saveSimpleListData(res[1], export_file + '.singularValue')
        #saveSimpleListData(res[2], export_file + '.rightSV')
        RV = np.transpose(RVt)
        U, S, V = np.flip(RU, axis=1), np.flip(RS), np.flip(RV, axis=1)
        return U, S, V


class TriangleCount:
    # arg mode: batch or incremental
    @staticmethod
    def THINKD(in_path, out_path, sampling_ratio, number_of_trials, mode):
        pass


class SeriesSummarization:
    @staticmethod
    def BEATLEX(data, param, out_path, name):
        beatlex = BeatLex(data.attrlists, param)
        result = beatlex.summarize_sequence()
        print('done')
        return result
