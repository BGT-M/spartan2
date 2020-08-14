#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: eaglemine
#  File: eaglemine.py
#  Goal: The main routine of eaglemine
#      Version:  1.0
#      Created by @wenchieh  on <12/17/2017>
#

__author__ = 'wenchieh'

# sys
import time

# third-party lib
import numpy as np
from scipy.sparse.linalg import svds

# project
from .eaglemine_model import EagleMineModel
from .tools.histogram_heuristic_generator import HistogramHeuristicGenerator

from .._model import DMmodel
from spartan.tensor.graph import Graph


class EagleMine( DMmodel ):
    ''' Micro-cluster detection: vision-guided anomaly detection.
    Given a histogram derived from the correlated features of graph nodes，
    EagleMine can be used to identify the micro-clusters in the graph,
    these nodes in micro-clusters basically corresponds to some anomaly patterns.
    '''
    def __init__(self, voctype:str="dtmnorm", mode:int=2, mix_comps:int=2):
        '''Initialization func for EagleMine
        Parameters:
        --------
        :param voctype: str
            vocabulary type: {"dtmnorm", "dmgauss"}
            Default is "dtmnorm".
        :param mode: int
            The dimensions of features (the histogram).
            Default is 2.
        :param mix_comps: int
            # mixture component for describing the major island.
            Default is 2.
        '''
        self.voctype = voctype
        self.mode = mode
        self.ncomps = mix_comps
        self.eaglemodel = None
        self.__histgenerator__ = None
    
    @classmethod
    def __create__(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    @staticmethod
    def get_graph_spectral(sparse_matrix):
        '''Extract the spectral features of the given sparse matrix (graph)
        Parameter:
        --------
        :param sparse_matrix:
            sparse matrix for adjacency matrix a graph.
        '''
        hub, _, auth = svds(1.0 * sparse_matrix.tocsr(), k=1, which='LM')
        hub, auth = np.squeeze(np.array(hub)), np.squeeze(np.array(auth))
        if abs(np.max(hub)) < abs(np.min(hub)):
            hub *= -1
        hub[hub < 0] = 0
        if abs(np.max(auth)) < abs(np.min(auth)):
            auth *= -1
        auth[auth < 0] = 0
        
        return hub, auth
    
    def graph2feature(self, graph:Graph, feature_type:str='outdegree2hubness'):
        '''Extract example correlated-features of the given graph and generate the corresponding histogram.
        Parameters:
        --------
        :param graph: Graph
            Given graph data
        :param feature_type: str
            Feature type for the graph node: {'outdegree2hubness', 'indegree2authority', 'degree2pagerank'}.
            Default is 'outdegree2hubness'.
        '''
        
        if feature_type not in {'outdegree2hubness', 'indegree2authority', 'degree2pagerank'}:
            raise NameError("Invalid feature type 'ty_type', which should be in {'outdegree2hubness', 'indegree2authority', 'degree2pagerank'}")
        
        print("extract graph features ..... ")
        start_tm = time.time()
        outd, ind = np.asarray(graph.sm.sum(axis=1)).squeeze(), np.asarray(graph.sm.sum(axis=0)).squeeze()
        hub, auth = self.get_graph_spectral(graph.sm)
        if feature_type == 'outdegree2hubness' or feature_type == 'degree2pagerank':
            deg, spec = outd, hub
        else:
            deg, spec = ind, auth
        degreeidx = 0
        print("done! @ {}s".format(time.time() - start_tm))
        
        return degreeidx, np.column_stack((deg, spec))
    
    def feature2histogram(self, feature:np.ndarray, degreeidx:int=0, 
                          N_bins:int=80, base:int=10, mode:int=2, verbose:bool=True):
        '''Construct histogram with given features
        Parameters:
        --------
        :param feature: np.ndarray
            The correlated features
        :param degreeidx: int
            The index of 'degree' ('out-degree', 'in-degree') in features, degreeidx=-1 if not containing 'degree' feature
            Default is 0.
        :param N_bins: int
            The expected number of bins for generating histogram.
            Default is 80.
        :param base: int
            The logarithmic base for bucketing the graph features.
            Default is 10.
        :param mode: int
            The dimensions of features for constructing the histogram.
            Default is 2.
        :param verbose: bool
            Whether output some running logs.
            Default is True.
        '''
        n_samples, n_features = feature.shape
        index = np.array([True] * n_samples)
        for mod in range(mode):
            index &= feature[:, mod] > 0
        if verbose:
            print("total shape: {}, valid samples:{}".format(feature.shape, np.sum(index)))
    
        degree, feat = None, None
        feature = feature[index, :]
        if degreeidx >= 0:
            degree = feature[:, degreeidx]
            feat = np.delete(feature, degreeidx, axis=1)
            del feature
        else:
            feat = feature
        
        print("construct histogram ..... ")
        start_tm = time.time()
        self.__histgenerator__ = HistogramHeuristicGenerator()
        if degree is not None:
            self.__histgenerator__.set_deg_data(degree, feat)
            self.__histgenerator__.histogram_gen(method="degree", N=N_bins, base=base)
        else:
            self.__histgenerator__.set_data(feat)
            self.__histgenerator__.histogram_gen(method="N", N=N_bins, logarithmic=True, base=base)
        print("done! @ {}s".format(time.time() - start_tm))
        
        if verbose:
            self.__histgenerator__.dump()
        
        n_nodes = len(self.__histgenerator__.points_coord)
        node2hcel = map(tuple, self.__histgenerator__.points_coord)
        nodeidx2hcel = dict(zip(range(n_nodes), node2hcel))
        
        return self.__histgenerator__.histogram, nodeidx2hcel, self.__histgenerator__.hpos2avgfeat
    
    def set_histdata(self, histogram:dict, node2hcel:dict, hcel2avgfeat:dict, weighted_ftidx:int=0):
        '''Set the histogram data
        Parameters:
        --------
        :param histogram: dict
            the format '(x,y,z,...): val', denoting that the cell (x,y,z,...) affiliates with value 'val'.
        :param node2hcel: dict
            graph node id (index) to histogram cell
        :param hcel2avgfeat: dict
            the average feature values for each histogram cell.
        :param weighted_ftidx: int
            The feature index as weight for suspiciousness metric.
            Default is 0.
        '''
        self.histogram, self.hcelsusp_wt = list(), list()
        
        for hcel in histogram.keys():
            self.histogram.append(list(hcel) + [histogram.get(hcel)])
            self.hcelsusp_wt.append(hcel2avgfeat[hcel][weighted_ftidx])
        
        self.histogram = np.asarray(self.histogram)
        self.hcelsusp_wt = np.asarray(self.hcelsusp_wt)
        self.node2hcel = np.column_stack((list(node2hcel.keys()), list(node2hcel.values())))
        self.eaglemodel = None
        self.__histgenerator__ = None
        self.hcel2label = None
        
    def run(self, outs:str, waterlevel_step:float=0.2, prune_alpha:float=0.80, 
            min_pts:int=20, strictness:int=3, verbose:bool=True):
        ''' micro-cluster identification and refinement with water-level tree.
        Parameters:
        --------
        :param outs: str
            Output path for some temporary results.
        :param waterlevel_step: float
            Step size for raising the water level.
            Default is 0.2.
        :param prune_alpha: float
            How proportion of pruning for level-tree.
            Default is 0.80.
        :param min_pts: int
            The minimum number of points in a histogram cell.
            Default is 20.
        :param strictness: int
            How strict should the anderson-darling test for normality. 0: not at all strict; 4: very strict
            Default is 3.
        :param verbose: bool
            Whether output some running logs.
            Default is True.
        '''
        print("*****************")
        print("[0]. initialization")
        self.eaglemodel = EagleMineModel(self.mode, self.ncomps)
        self.eaglemodel.set_vocabulary(self.voctype)
        self.eaglemodel.set_histogram(self.histogram)
        
        print("*****************")
        print("[1]. WaterLevelTree")
        start_tm = time.time()
        self.eaglemodel.leveltree_build(outs, waterlevel_step, prune_alpha, verbose=verbose)
        end_tm1 = time.time()
        print("done @ {}".format(end_tm1 - start_tm))
        
        print("*****************")
        print("[2]. TreeExplore")
        self.eaglemodel.search(min_pts, strictness, verbose=verbose)
        self.eaglemodel.post_stitch(strictness, verbose=verbose)
        end_tm2 = time.time()
        print("done @ {}".format(end_tm2 - end_tm1))
        
        print("*****************")
        print("[3]. node groups cluster and suspicious measure")
        self.hcel2label, mdl = self.eaglemodel.cluster_remarks(strictness, verbose=verbose)
        cluster_suspicious = self.eaglemodel.cluster_weighted_suspicious(self.hcelsusp_wt, strictness, verbose=verbose)
        
        # print("description length (mdl): {}".format(mdl))
        # print("suspicious result: {}".foramt(cluster_suspicious))
        
        print('done @ {}'.format(time.time() - start_tm))
    
    def __str__(self):
        return str(vars(self))

    def dump(self):
        self.eaglemodel.dump()
        if self.__histgenerator__ is not None:
            self.__histgenerator__.dump()
        print("done!")

    def save(self, outfn_eaglemine:str, outfn_leveltree:str=None, 
             outfn_node2label:str=None, outfn_hcel2label:str=None, 
             comments:str="#", delimiter:str=";"):
        '''save result of EagleMine
        Parameters:
        --------
        :param outfn_eaglemine: str
            Output path for eaglemine data
        :param outfn_leveltree: str
            Output path for the eater-level-tree data.
        :param outfn_node2label: str
            Output path for node2label data
        :param outfn_hcel2label: str
            Output path for hcel2label data
        :param comments: str
            The comments (start character) of inputs.
            Default is "#".
        :param delimiter: str
            The separator of items in each line of inputs.
            Default is ";".
        '''
        print("saving result")
        start_tm = time.time()
        self.eaglemodel.save(outfn_eaglemine)
        if outfn_leveltree:
            self.eaglemodel.leveltree.save_leveltree(outfn_leveltree, verbose=False)
        
        nlabs = len(np.unique(list(self.hcel2label.values())))
        if outfn_node2label is not None:
            nnds = len(self.node2hcel)
            with open(outfn_node2label, 'w') as ofp_node2lab:
                ofp_node2lab.writelines(comments + " #pt: {}, #label: {}\n".format(nnds, nlabs))
                for k in range(nnds):
                    nodeidx, hcel = self.node2hcel[k, 0], tuple(self.node2hcel[k, 1:])
                    nodelab = self.hcel2label.get(hcel, -1)
                    ofp_node2lab.writelines("{}{}{}\n".format(nodeidx, delimiter, nodelab))
                ofp_node2lab.close()
                
        if outfn_hcel2label is not None:
            nhcels = len(self.hcelsusp_wt)
            with open(outfn_hcel2label, 'w') as ofp_hcel2lab:
                ofp_hcel2lab.writelines(comments + ' #hcel: {}, #label: {}'.format(nhcels, nlabs))
                for hcel, lab in self.hcel2label.items():
                    hcel_str = delimiter.join(map(str, hcel))
                    ofp_hcel2lab.writelines("{}{}{}\n".format(hcel_str, delimiter, lab))
                ofp_hcel2lab.close()

        end_tm = time.time()        
        print("done! @ {}s".format(end_tm - start_tm))
    
    def save_histogram(self,outfn_histogram:str=None, outfn_node2hcel:str=None, outfn_hcel2avgfeat:str=None, 
                            comments:str="#", delimiter:str=","):
        '''Save the histogram data for the graph.
        Parameters:
        --------
        :param outfn_histogram: str
            Output path of histogram.
            The record in histogram should be in the format 'x,y,z,...,val', denoting that the cell (x, y, z, ...) affiliates with value 'val'
            Default is None.
        :param outfn_node2hcel: str
            Output path of the file mapping the node to histogram cell.
            Default is None.
        :param outfn_hcel2avgfeat: str
            Output path of the file mapping the histogram cell to the average features and #points
            Default is None.
        :param comments: str
            The comments (start character) of inputs.
            Default is "#".
        :param delimiter: str
            The separator of items in each line of inputs.
            Default is ",".
        '''
        if self.__histgenerator__ is not None:
            if outfn_histogram is not None:
                self.__histgenerator__.save_histogram(outfn_histogram, delimiter, comments)
            if outfn_node2hcel is not None:
                self.__histgenerator__.save_pts_index(outfn_node2hcel, delimiter, comments)
            if outfn_hcel2avgfeat is not None:
                self.__histgenerator__.save_hpos2avgfeat(outfn_hcel2avgfeat, delimiter, comments)
        else:
            RuntimeError("No histogram generated for given graph!")
    
    ## TODO: relized the 'anomaly_detection' for the task based running. The root API may need to be refactored.
    # def anomaly_detection(self, outs:str, waterlevel_step:float=0.2, 
    #                       prune_alpha:float=0.80, min_pts:int=20, strictness:int=3):
    #     '''anomaly detection with EagleMine
    #     Parameters:
    #     --------
    #     outs: str
    #         Output path for some temporary results.
    #     waterlevel_step: float
    #         Step size for raising the water level.
    #         Default is 0.2.
    #     prune_alpha: float
    #         How proportion of pruning for level-tree.
    #         Default is 0.80.
    #     min_pts: int
    #         The minimum number of points in a histogram cell.
    #         Default is 20.
    #     strictness: int
    #         How strict should the anderson-darling test for normality. 0: not at all strict; 4: very strict
    #         Default is 3.
    #     '''
    #     return self.run(outs, waterlevel_step, prune_alpha, min_pts, strictness, True)
    
