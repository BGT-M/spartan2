import numpy as np
from .mytools.MinTree import MinTree

from .._model import DMmodel
from ...util.basicutil import param_default
import copy
from collections import defaultdict
from .util import get_real_res, preprocess_data

class CubeFlow(DMmodel):
    '''Anomaly detection base on contrastively dense subgraphs, considering
    topological, temporal, and categorical (e.g. rating scores) signals, or
    any supported combinations.

    Parameters
    ----------
    graph: Graph
        Graph instance contains adjency matrix, and possible multiple signals.
    '''
    def __init__(self, stensorList:list, **params):
        self.alpha = param_default(params, 'alpha', 0.8)
        self.k = param_default(params, 'k', 1)
        self.dim = param_default(params, 'dim', 3)
        self.outpath = param_default(params, 'outpath', './output/CFD-3/')
        self.graphlist, self.mt_dictList = preprocess_data(stensorList, self.dim)
        self.graphnum = len(self.graphlist)
        
    def __str__(self):
        return str(vars(self))
    
    def anomaly_detection(self):
        return self.run()

    def run(self):
        print("you are running with ", self.graphnum + 1, "partite graph")
        self.nres = []
        self.initData()
        for i in range(self.k):
            finalsets, score = self.fastGreedyDecreasing()
            real_res = get_real_res([finalsets, score], self.dim, self.mt_dictList, self.outpath)
            self.nres.append(real_res)
            self.del_block(finalsets)
        return self.nres

    def initData(self):
        AM_tensor_data = self.graphlist[0].graph_tensor._data.copy()  # A, M, T, money
        CM_tensor_data = self.graphlist[1].graph_tensor._data.copy()  # C, M, T, money
        self.AM_mat = AM_tensor_data.to_scipy_sparse().tolil()  # A,(M,T),money
        self.CM_mat = CM_tensor_data.to_scipy_sparse().tolil()  # C,(M,T),money
        self.AM_tran_mat = AM_tensor_data.to_scipy_sparse().T.tolil()  # A,(M,T),money
        self.CM_tran_mat = CM_tensor_data.to_scipy_sparse().T.tolil()  # C,(M,T),money
        print(self.AM_mat.shape, self.CM_mat.shape)

    # find the smallest one in all set
    def findmin(self):
        min_tree_i = -1
        min_indices = -1
        min_value = float('inf')
        for i in range(len(self.dtrees)):
            index, value = self.dtrees[i].getMin()
            if value < min_value:
                min_value = value
                min_tree_i = i
                min_indices = index
    
        return min_tree_i, min_indices

    def checkset(self):
        res = True
        for i in range(len(self.sets)):
            if len(self.sets[i]) == 0:
                res = False
                break
        return res

    def initGreedy(self):
        self.sets = []
        self.dtrees = []
        self.mid_min = []
        self.midDeltas1 = []
        self.midDeltas2 = []
        self.curScore1 = 0  # sum of f
        self.curScore2 = 0  # sum of (q-f)
        self.bestAveScore = 0
        self.bestNumDeleted = defaultdict(int)
        
        rowDeltas = np.squeeze(self.AM_mat.sum(axis=1).A)  # sum of A
        self.dtrees.append(MinTree(rowDeltas))  # tree 0: u
        
        self.midDeltas1, self.midDeltas2 = np.array(np.squeeze(self.AM_tran_mat.sum(axis=1).A)), \
                                           np.array(np.squeeze(self.CM_tran_mat.sum(axis=1).A))
        difflen = abs(len(self.midDeltas1) - len(self.midDeltas2))
        if len(self.midDeltas1) > len(self.midDeltas2):
            self.midDeltas2 = np.append(self.midDeltas2, np.zeros((1, difflen)))
        if len(self.midDeltas1) < len(self.midDeltas2):
            self.midDeltas1 = np.append(self.midDeltas1, np.zeros((1, difflen)))
        print(len(self.midDeltas1), len(self.midDeltas2))
        
        mid_min, mid_max = [], []
        for m1, m2 in zip(self.midDeltas1, self.midDeltas2):
            mid_min.append(min(m1, m2))  # fi
            mid_max.append(max(m1, m2))  # qi
        self.mid_min, self.mid_max = np.array(mid_min), np.array(mid_max)
        mid_priority = self.mid_min - self.alpha * self.mid_max  # f- alpha * q
        self.dtrees.append(MinTree(mid_priority))  # tree 1: (v,t)
        
        colDeltas = np.squeeze(self.CM_mat.sum(axis=1).A)  # sum of C
        self.dtrees.append(MinTree(colDeltas))  # tree 2: w

        self.curScore1 = sum(mid_min) 
        self.curScore2 = sum(abs(self.midDeltas1 - self.midDeltas2))
        
        self.sets.append(set(range(self.AM_mat.shape[0])))
        self.sets.append(set(range(len(self.mid_min))))
        self.sets.append(set(range(self.CM_mat.shape[0])))
        s = sum([len(self.sets[i]) for i in range(len(self.sets))])
        print('size: ', s)
        
        curAveScore = ((1 - self.alpha) * self.curScore1 - self.alpha * self.curScore2) / s
        print('initail score of g: ', curAveScore)
        self.bestAveScore = curAveScore
        
    def updataConnNode(self, min_tree_i, idx):
        if min_tree_i == 0:  # delete a node from u
            # update  the  weight of connected nodes
            for j in self.AM_mat.rows[idx]: 
                if self.midDeltas1[j] <= 0:
                    continue
                self.curScore1 -= self.mid_min[j] 
                self.curScore2 -= abs(self.midDeltas1[j] - self.midDeltas2[j])
                
                self.midDeltas1[j] -= self.AM_mat[idx, j]
                f, q = min(self.midDeltas1[j], self.midDeltas2[j]), max(self.midDeltas1[j], self.midDeltas2[j])
                self.mid_min[j] = f
                self.curScore1 += f
                self.curScore2 += (q - f)
                
                new_mid_w = f - self.alpha * q
                self.dtrees[1].setVal(j, new_mid_w)
        elif min_tree_i == 2:  # delete a node from w
            # update mD1, mD2, and mid_tree
            for j in self.CM_mat.rows[idx]:
                if self.midDeltas2[j] <= 0:
                    continue
                self.curScore1 -= self.mid_min[j]
                self.curScore2 -= abs(self.midDeltas1[j] - self.midDeltas2[j])
                
                self.midDeltas2[j] -= self.CM_mat[idx, j]
                f, q = min(self.midDeltas1[j], self.midDeltas2[j]), max(self.midDeltas1[j], self.midDeltas2[j])
                self.mid_min[j] = f
                self.curScore1 += f
                self.curScore2 += (q - f)

                new_mid_w = f - self.alpha * q
                self.dtrees[1].setVal(j, new_mid_w)
        elif min_tree_i == 1:  # delete a node from M
            if idx < self.AM_tran_mat.shape[0]:
                for j in self.AM_tran_mat.rows[idx]:
                    # if self.dtrees[0].index_of(j) <= 0:
                    #     continue
                    self.dtrees[0].changeVal(j, -self.AM_tran_mat[idx, j])
            if idx < self.CM_tran_mat.shape[0]:
                for j in self.CM_tran_mat.rows[idx]:  # update connected nodes in C
                    # if self.dtrees[2].index_of(j) <= 0:
                    #     continue
                    self.dtrees[2].changeVal(j, -self.CM_tran_mat[idx, j])
            
            self.curScore1 -= self.mid_min[idx]
            self.curScore2 -= abs(self.midDeltas1[idx] - self.midDeltas2[idx])
            self.midDeltas1[idx] = -1
            self.midDeltas2[idx] = -1
        self.sets[min_tree_i] -= {idx}
        self.dtrees[min_tree_i].changeVal(idx, float('inf'))
        self.deleted[min_tree_i].append(idx)
        self.numDeleted[min_tree_i] += 1
    
    def fastGreedyDecreasing(self):
        print('this is the cpu version of CubeFlow')  
        print('start  greedy')

        self.initGreedy()
        self.numDeleted = defaultdict(int)
        self.deleted = {}
        for i in range(len(self.sets)):
            self.deleted[i] = []
        finalsets = copy.deepcopy(self.sets)
        # repeat deleting until one node set is empty
        while self.checkset(): 
            min_tree_i, idx = self.findmin()                 
            self.updataConnNode(min_tree_i, idx)
            s = sum([len(self.sets[i]) for i in range(len(self.sets))])
            curAveScore = ((1 - self.alpha) * self.curScore1 - self.alpha * self.curScore2) / s
            if curAveScore >= self.bestAveScore:
                for i in range(len(self.sets)):
                    self.bestNumDeleted[i] = self.numDeleted[i]
                self.bestAveScore = curAveScore
        print('best delete number : ', self.bestNumDeleted)
        print('nodes number remaining:  ', len(self.sets[0]), len(self.sets[1]), len(self.sets[2]))
        print('best score of g(S): ', self.bestAveScore)  
        for i in range(len(finalsets)):
            best_deleted = self.deleted[i][:self.bestNumDeleted[i]]
            finalsets[i] = finalsets[i] - set(best_deleted)
        print('size of found subgraph:  ', len(finalsets[0]), len(finalsets[1]), len(finalsets[2]))
        return finalsets, self.bestAveScore

    def del_block(self, finalsets):
        a_set, mt_set, c_set = finalsets[0], finalsets[1], finalsets[2]
        a_mt_nnz = self.AM_mat.nonzero()
        for a, mt in zip(a_mt_nnz[0], a_mt_nnz[1]):
            if a in a_set and mt in mt_set:
                self.AM_mat[a, mt] = 0
                self.AM_tran_mat[mt, a] = 0
        c_mt_nnz = self.CM_mat.nonzero()
        for c, mt in zip(c_mt_nnz[0], c_mt_nnz[1]):
            if c in c_set and mt in mt_set:
                self.CM_mat[c, mt] = 0
                self.CM_tran_mat[mt, c] = 0
