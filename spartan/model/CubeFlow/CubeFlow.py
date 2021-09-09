import numpy as np
from .mytools.MinTree import MinTree

from .._model import DMmodel
from ...util.basicutil import param_default
import copy
from collections import defaultdict
from .util import get_real_res, preprocess_data, loadtxt2res, saveres2txt, get_zero_matrix
import pdb

import sparse as sp

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
        self.graphlist, self.mt_dictList, self.m_mt_dict, self.m_mtSize_dict_ori = preprocess_data(stensorList, self.dim)
        self.graphnum = len(self.graphlist)
        
    def __str__(self):
        return str(vars(self))
    
    def anomaly_detection(self):
        return self.run()

    def run(self, del_type=0, maxsize=(-1,-1,-1), is_find_all_blocks=False):
        print("you are running with ", self.graphnum + 1, "partite graph")
        self.del_type = del_type
        self.nres = []
        self.size_max_limit = maxsize
        self.has_limit = False
        self.is_find_all_blocks = is_find_all_blocks
        self.m_mtSize_dict_delete = copy.deepcopy(self.m_mtSize_dict_ori)
        
        if type(self.size_max_limit) == int and self.size_max_limit != -1:
            print('Block size sum (X+Y+Z) limit:', self.size_max_limit)
            self.has_limit = True
            
        elif type(self.size_max_limit) == tuple:
            if len(self.size_max_limit) != 3:
                raise Exception('The length of size_max_limit tuple should be 3!')
            
            if not (self.size_max_limit[0] == -1 and self.size_max_limit[1] == -1 and self.size_max_limit[2] == -1):
                print(f'Block size limits: (X:{self.size_max_limit[0]}, Y:{self.size_max_limit[1]}, Z:{self.size_max_limit[2]})')
                self.has_limit = True
        else:
            print('No block size limit.')
        
        self.initData()
        if not is_find_all_blocks:
            for i in range(self.k):
                finalsets, score = self.fastGreedyDecreasing()
                real_res = get_real_res([finalsets, score], self.dim, self.mt_dictList, self.outpath)
                        
                self.nres.append(real_res)
                self.del_block(finalsets)
        else:
            if self.del_type == 0:
                raise ValueError("You cannot find all blocks with del_type equals to 0!")
            
            if self.has_limit:
                curr_i = 0
                while self.checkset(self.sets_ori):
                    finalsets, score = self.fastGreedyDecreasing()
                    real_res = get_real_res([finalsets, score], self.dim, self.mt_dictList, self.outpath)
                    
                    self.nres.append(real_res)
                    self.del_block(finalsets)
                    curr_i += 1
            else:
                print('ERROR: You cannot find all blocks without specifying the maximum size of the block!')
                return 
        return self.nres

    def initData(self):
        AM_tensor_data = self.graphlist[0].graph_tensor._data.copy()  # A, M, T, money
        CM_tensor_data = self.graphlist[1].graph_tensor._data.copy()  # C, M, T, money
        self.AM_mat = AM_tensor_data.to_scipy_sparse().tolil()  # A,(M,T),money
        self.CM_mat = CM_tensor_data.to_scipy_sparse().tolil()  # C,(M,T),money
        self.AM_tran_mat = AM_tensor_data.to_scipy_sparse().T.tolil()  # A,(M,T),money
        self.CM_tran_mat = CM_tensor_data.to_scipy_sparse().T.tolil()  # C,(M,T),money
        print(self.AM_mat.shape, self.CM_mat.shape)
        
        if self.has_limit:
            self.m_set_ori = set(list(self.m_mt_dict.keys()))
            print('m size:', len(self.m_set_ori))
        else:
            self.m_set_ori = set()
        
        self.sets_ori = []
        if self.is_find_all_blocks:
            midDeltas1, midDeltas2 = np.array(np.squeeze(self.AM_tran_mat.sum(axis=1).A)), \
                                            np.array(np.squeeze(self.CM_tran_mat.sum(axis=1).A))
            
            self.sets_ori.append(set(range(self.AM_mat.shape[0])))
            self.sets_ori.append(set(range(max(len(midDeltas1), len(midDeltas2)))))
            self.sets_ori.append(set(range(self.CM_mat.shape[0])))
            print(len(self.sets_ori[0]), len(self.sets_ori[1]), len(self.sets_ori[2]))

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

    def checkset(self, sets):
        res = True
        for i in range(len(sets)):
            if len(sets[i]) == 0:
                res = False
                break
        return res
    
    def checkset_size_max_limit(self):
        now_x_size, now_y_size, now_z_size = len(self.sets[0]), len(self.m_set), len(self.sets[2])
        
        if type(self.size_max_limit) == int:
            if (now_x_size + now_y_size + now_z_size) > self.size_max_limit:
                return False
            
        elif type(self.size_max_limit) == tuple:
            x_max, y_max, z_max = self.size_max_limit
            if x_max != -1 and now_x_size > x_max:
                return False
            if y_max != -1 and now_y_size > y_max:
                return False
            if z_max != -1 and now_z_size > z_max:
                return False
            
        return True

    def initGreedy(self):
        self.sets = []
        self.dtrees = []
        self.mid_min = []
        self.midDeltas1 = []
        self.midDeltas2 = []
        self.curScore1 = 0  # sum of f
        self.curScore2 = 0  # sum of (q-f)
        self.bestAveScore = 0
        self.bestAveSocre1 = 0 # sum of f when algorithm gets best score
        self.bestAveSocre2 = 0 # sum of (q-f) when algorithm gets best score
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
        print('s size: ', s)
        
        if self.has_limit:
            self.m_mtSize_dict = copy.deepcopy(self.m_mtSize_dict_ori)
            self.m_set = set(list(self.m_mt_dict.keys()))
        
        curAveScore = float('-inf') # 看一下是否更改把numDeleted的初始值
        
        if self.has_limit:
            if self.checkset_size_max_limit():
                curAveScore = ((1 - self.alpha) * self.curScore1 - self.alpha * self.curScore2) / s
        else:
            curAveScore = ((1 - self.alpha) * self.curScore1 - self.alpha * self.curScore2) / s
            
        print('initial score of g: ', curAveScore)
        self.bestAveScore = curAveScore
        self.bestAveSocre1 = self.curScore1
        self.bestAveSocre2 = self.curScore2
        
    def updateConnNode(self, min_tree_i, idx):
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
                    self.dtrees[0].changeVal(j, -self.AM_tran_mat[idx, j])
            if idx < self.CM_tran_mat.shape[0]:
                for j in self.CM_tran_mat.rows[idx]:  # update connected nodes in C
                    self.dtrees[2].changeVal(j, -self.CM_tran_mat[idx, j])
            
            self.curScore1 -= self.mid_min[idx]
            self.curScore2 -= abs(self.midDeltas1[idx] - self.midDeltas2[idx])
            self.midDeltas1[idx] = -1
            self.midDeltas2[idx] = -1
        self.sets[min_tree_i] -= {idx}
        self.dtrees[min_tree_i].changeVal(idx, float('inf'))
        self.deleted[min_tree_i].append(idx)
        self.numDeleted[min_tree_i] += 1
        
        if self.has_limit and min_tree_i == 1: # remove corresponding m in m_set
            # map mt to m
            tmp_m = self.mt_dictList[0][idx]
            self.m_mtSize_dict[tmp_m] -= 1
            
            if self.m_mtSize_dict[tmp_m] <= 0: # all mt have been removed
                self.m_set -= {tmp_m}
    
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
        while self.checkset(self.sets): 
            min_tree_i, idx = self.findmin()
            self.updateConnNode(min_tree_i, idx)
            s = sum([len(self.sets[i]) for i in range(len(self.sets))])
            curAveScore = ((1 - self.alpha) * self.curScore1 - self.alpha * self.curScore2) / s
                
            if self.has_limit:
                if curAveScore >= self.bestAveScore and self.checkset_size_max_limit() and self.checkset(self.sets):
                    for i in range(len(self.sets)):
                        self.bestNumDeleted[i] = self.numDeleted[i]
                    self.bestAveScore = curAveScore
                    self.bestAveSocre1 = self.curScore1
                    self.bestAveSocre2 = self.curScore2
            else:
                if curAveScore >= self.bestAveScore and self.checkset(self.sets):
                    for i in range(len(self.sets)):
                        self.bestNumDeleted[i] = self.numDeleted[i]
                    self.bestAveScore = curAveScore
                    self.bestAveSocre1 = self.curScore1
                    self.bestAveSocre2 = self.curScore2
                
        
        print('best delete number : ', self.bestNumDeleted)
        print('nodes number remaining:  ', len(self.sets[0]), len(self.sets[1]), len(self.sets[2]))
        print(f'best score of g(S): {self.bestAveScore}, f: {self.bestAveSocre1}, q-f: {self.bestAveSocre2}')
        
        for i in range(len(finalsets)):
            best_deleted = self.deleted[i][:self.bestNumDeleted[i]]
            finalsets[i] = finalsets[i] - set(best_deleted)
        print('size of found subgraph:  ', len(finalsets[0]), len(finalsets[1]), len(finalsets[2]))
        
        if sum(self.bestNumDeleted.values()) == 0:  
            # 没有找到任何子图满足限制条件，返回原始子图。例如，限制了m的大小，算法把a集合删空了之后m大小依然大于限制。
            print('Do not find any sub-block that meets the settings.')
            for i in range(len(finalsets)):
                finalsets[i] = copy.deepcopy(self.sets_ori[i])
                
            return finalsets, float('-inf')
        
        return finalsets, self.bestAveScore

    def del_block(self, finalsets):
        a_set, mt_set, c_set = set(finalsets[0]), set(finalsets[1]), set(finalsets[2])
        
        if self.del_type == 0: # del find edges
            a_mt_nnz = self.AM_mat.nonzero()
            for a, mt in zip(a_mt_nnz[0], a_mt_nnz[1]):
                if a in a_set and mt in mt_set:
                    self.AM_mat[a, mt] = 0
                    self.AM_tran_mat[mt, a] = 0
                    ##################
                    if self.has_limit:
                        tmp_m = self.mt_dictList[0][mt]
                        self.m_mtSize_dict_delete[tmp_m] -= 1
                        if self.m_mtSize_dict_delete[tmp_m] <= 0:  # all mt have been removed
                            self.m_set_ori -= {tmp_m}
                    
            c_mt_nnz = self.CM_mat.nonzero()
            for c, mt in zip(c_mt_nnz[0], c_mt_nnz[1]):
                if c in c_set and mt in mt_set:
                    self.CM_mat[c, mt] = 0
                    self.CM_tran_mat[mt, c] = 0
                    ###############
                    if self.has_limit:
                        tmp_m = self.mt_dictList[0][mt]
                        self.m_mtSize_dict_delete[tmp_m] -= 1
                        if self.m_mtSize_dict_delete[tmp_m] <= 0:  # all mt have been removed
                            self.m_set_ori -= {tmp_m}
                    
        elif self.del_type == 1: # del find points
            # map mt => m
            m_set = []
            for mt in mt_set:
                m_set.append(self.mt_dictList[0][mt])
            m_set = set(m_set)
            print('find m len:', len(m_set))
            
            # find all mt
            total_mt_set = []
            for m in m_set:
                total_mt_set.extend(self.m_mt_dict[m])
            total_mt_set = set(total_mt_set)
            
            if self.has_limit:
                self.m_set_ori -= m_set
            
            if self.is_find_all_blocks:
                self.sets_ori[0] -= a_set
                self.sets_ori[1] -= total_mt_set
                self.sets_ori[2] -= c_set
            
            a_mt_nnz = self.AM_mat.nonzero()
            for a, mt in zip(a_mt_nnz[0], a_mt_nnz[1]):
                if a in a_set or mt in total_mt_set:
                    self.AM_mat[a, mt] = 0
                    self.AM_tran_mat[mt, a] = 0

            c_mt_nnz = self.CM_mat.nonzero()
            for c, mt in zip(c_mt_nnz[0], c_mt_nnz[1]):
                if c in c_set or mt in total_mt_set:
                    self.CM_mat[c, mt] = 0
                    self.CM_tran_mat[mt, c] = 0   
                    
        elif self.del_type == 2:
            m_set = []
            for mt in mt_set:
                m_set.append(self.mt_dictList[0][mt])
            m_set = set(m_set)
            print('find m len:', len(m_set))

            # find all mt
            total_mt_set = []
            for m in m_set:
                total_mt_set.extend(self.m_mt_dict[m])
            total_mt_set = set(total_mt_set)

            if self.has_limit:
                self.m_set_ori -= m_set

            if self.is_find_all_blocks:
                self.sets_ori[0] -= a_set
                self.sets_ori[1] -= total_mt_set
                self.sets_ori[2] -= c_set
                
            a_zero_size, mt_zero_size = self.AM_mat.shape
            c_zero_size, _ = self.CM_mat.shape
            A_zero_mat = get_zero_matrix(a_zero_size, list(a_set))
            MT_zero_mat = get_zero_matrix(mt_zero_size, list(total_mt_set))
            C_zero_mat = get_zero_matrix(c_zero_size, list(c_set))
            
            self.AM_mat = A_zero_mat * self.AM_mat * MT_zero_mat
            self.AM_tran_mat = MT_zero_mat * self.AM_tran_mat * A_zero_mat
            
            self.CM_mat = C_zero_mat * self.CM_mat * MT_zero_mat
            self.CM_tran_mat = MT_zero_mat * self.CM_tran_mat * C_zero_mat
            
        else:
            raise Exception('Unknown del_type! (del_type should be in [0, 1])')

    @classmethod
    def loadtxt2res(cls, *args, **kwargs):
        return loadtxt2res(*args, **kwargs)

    @classmethod
    def saveres2txt(cls, *args, **kwargs):
        return saveres2txt(*args, **kwargs)
