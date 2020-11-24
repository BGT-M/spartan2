import sys, os, time
import numpy as np
from .mytools.MinTree import MinTree

from .._model import DMmodel
from ...util.basicutil import param_default
from ...backend import STensor

def del_block(M, rowSet ,colSet):
    M = M.tolil()

    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()


class FlowScope( DMmodel ):
    '''Anomaly detection base on contrastively dense subgraphs, considering
    topological, temporal, and categorical (e.g. rating scores) signals, or
    any supported combinations.

    Parameters
    ----------
    graph: Graph
        Graph instance contains adjency matrix, and possible multiple signals.
    '''
    def __init__(self, graphList: list, **params):
        self.graphnum = len(graphList)
        self.graphlist = graphList
        # self.alpha = param_default(params, 'alpha', 0.8)
        # self.alg = param_default(params, 'alg', 'fastgreedy')

    def __str__(self):
        return str(vars(self))

    
    def run(self, k:int=3, level:int=0, alpha:float=4):
        print("you are running with ", self.graphnum+1," partite graph")
        self.level = level
        self.alpha = alpha
        self.initData()
        self.nres = []


        for i in range(k):
            if self.level == 0:
                (finalsets, score) = self.fastGreedyDecreasing()
            else:
                return print("No such level know")

            self.nres.append([finalsets, score])

            for j in range(len(self.mcurlist)):
                self.mcurlist[j] = del_block(self.mcurlist[j], finalsets[j], finalsets[j+1])
                self.mtranslist[j] = del_block(self.mtranslist[j], finalsets[j+1], finalsets[j])


        return self.nres


    def initData(self):
        self.mcurlist = []
        self.mtranslist = []
        for i in range(len(self.graphlist)):
            self.mcurlist.append(self.graphlist[i].graph_tensor._data.copy().tocsr().tolil().astype(np.float64))
            self.mtranslist.append(self.graphlist[i].graph_tensor._data.copy().tocsr().tolil().transpose()) 


    def initGreedy(self):
        self.sets = []
        self.dtrees = []
        self.deltaslist = []
        self.curScorelist = []
        self.curAveScorelist = []



        self.sets.append(set(range(self.mcurlist[0].shape[0])))
        for i in range(len(self.mcurlist)-1):
            self.sets.append(set(range(self.mcurlist[i].shape[1])))
        self.sets.append(set(range(self.mcurlist[-1].shape[1])))

        s = 0
        for i in range(len(self.sets)):
            s += len(self.sets[i])
        
        rowDeltas = np.squeeze(self.mcurlist[0].sum(axis=1, dtype=np.float64).A)  # sum of A
        self.dtrees.append(MinTree(rowDeltas))
        for i in range(len(self.mcurlist)-1):
            midDeltas1 = np.squeeze(self.mcurlist[i].sum(axis=0, dtype=np.float64).A)
            midDeltas2 = np.squeeze(self.mcurlist[i+1].sum(axis=1, dtype=np.float64).A)
            self.deltaslist.append(midDeltas1)
            self.deltaslist.append(midDeltas2)
            
            mid_min = []
            mid_max = []
            for (m1, m2) in zip(midDeltas1, midDeltas2):
                temp = min(m1, m2)  # fi
                temp2 = max(m1, m2)  # qi
                mid_min.append(temp)
                mid_max.append(temp2)

            mid_min = np.array(mid_min)
            mid_max = np.array(mid_max)
            mid_priority = (1 + self.alpha) * mid_min - self.alpha * mid_max
            self.dtrees.append(MinTree(mid_priority))

            curScore1 = sum(mid_min)
            curScore2 = sum(abs(midDeltas1 - midDeltas2))
            self.curScorelist.append(curScore1)
            self.curScorelist.append(curScore2)
            curAveScore1 = curScore1 / s
            curAveScore2 = curScore2 / s
            self.curAveScorelist.append(curAveScore1)
            self.curAveScorelist.append(curAveScore2)

        colDeltas = np.squeeze(self.mcurlist[-1].sum(axis=0, dtype=np.float64).A)  # sum of C
        self.dtrees.append(MinTree(colDeltas))
        
        

    def updataConnNode(self, mold, index, mindelta):
        if mold == 0:
            # update  the  weight of connected nodes
            for j in self.mcurlist[0].rows[mindelta]:
                
                if self.deltaslist[0][j] == -1:
                    continue
                new_md1 = self.deltaslist[0][j] - self.mcurlist[0][mindelta, j]
                tempmin = min(self.deltaslist[0][j], self.deltaslist[1][j])
                if (new_md1) < tempmin:  # if in-degree know is the smallist one, update the curScore1
                    self.curScorelist[0] -= (tempmin - new_md1)

                self.curScorelist[1] += abs(new_md1 - self.deltaslist[1][j]) - abs(self.deltaslist[0][j] - self.deltaslist[1][j])
                self.deltaslist[0][j] = new_md1

                # update mid_tree
                mid_delta_value = abs(self.deltaslist[0][j] - self.deltaslist[1][j])
                new_mid_w = min(self.deltaslist[0][j], self.deltaslist[1][j]) - self.alpha * mid_delta_value
                self.dtrees[1].setVal(j, new_mid_w)

            self.sets[0] -= {mindelta}  # update rowSet
            self.dtrees[0].changeVal(mindelta, float('inf'))
            self.deleted.append((index, mindelta)) 
            self.numDeleted += 1

        elif mold == 2:
            # update mD1, mD2, and mid_tree
            for i in self.mtranslist[-1].rows[mindelta]:
                if self.deltaslist[-1][i] == -1:
                    continue

                new_md2 = self.deltaslist[-1][i] - self.mtranslist[-1][mindelta, i]
                tempmin = min(self.deltaslist[-1][i], self.deltaslist[-2][i])
                if (new_md2) < tempmin:
                    self.curScorelist[-2] -= (tempmin - new_md2)

                self.curScorelist[-1] += abs(new_md2 - self.deltaslist[-2][i]) - abs(self.deltaslist[-1][i] - self.deltaslist[-2][i])
                self.deltaslist[-1][i] = new_md2

                mid_delta_value = abs(self.deltaslist[-1][i] - self.deltaslist[-2][i])
                new_mid_w = min(self.deltaslist[-1][i], self.deltaslist[-2][i]) - self.alpha * mid_delta_value
                self.dtrees[-2].setVal(i, new_mid_w)

            self.sets[-1] -= {mindelta}
            self.dtrees[-1].changeVal(mindelta, float('inf'))
            self.deleted.append((index, mindelta))
            self.numDeleted += 1

        elif mold == 1:

            self.curScorelist[2 * index - 2] -= min(self.deltaslist[2 * index - 2][mindelta],
                                                    self.deltaslist[2 * index - 1][mindelta])
            self.curScorelist[2 * index - 1] -= abs(
                self.deltaslist[2 * index - 2][mindelta] - self.deltaslist[2 * index - 1][mindelta])

            self.deltaslist[2 * index - 2][mindelta] = -1
            self.deltaslist[2 * index - 1][mindelta] = -1

            for j in self.mcurlist[index].rows[mindelta]:
                if index < (len(self.dtrees) - 2):
                    if self.deltaslist[2 * index][j] == -1:
                        continue
                    new_md1 = self.deltaslist[2 * index][j] - self.mcurlist[index][mindelta, j]
                    tempmin = min(self.deltaslist[2 * index][j], self.deltaslist[2 * index + 1][j])
                    if (new_md1) < tempmin:  # if in-degree know is the smallist one, update the curScore1
                        self.curScorelist[2 * index] -= (tempmin - new_md1)

                    self.curScorelist[2 * index + 1] += abs(new_md1 - self.deltaslist[2 * index + 1][j]) - abs(
                        self.deltaslist[2 * index][j] - self.deltaslist[2 * index + 1][j])
                    self.deltaslist[2 * index][j] = new_md1


                    mid_delta_value = abs(self.deltaslist[2 * index][j] - self.deltaslist[2 * index + 1][j])
                    new_mid_w = min(self.deltaslist[2 * index][j], self.deltaslist[2 * index + 1][j]) - self.alpha * mid_delta_value
                    self.dtrees[index + 1].setVal(j, new_mid_w)
                else:
                    self.dtrees[index + 1].changeVal(j, -self.mcurlist[index][mindelta, j])

            for j in self.mtranslist[index - 1].rows[mindelta]:
                if index > 1:
                    if self.deltaslist[2 * index - 3][j] == -1:
                        continue

                    new_md2 = self.deltaslist[2 * index - 3][j] - self.mtranslist[index - 1][mindelta, j]
                    tempmin = min(self.deltaslist[2 * index - 3][j], self.deltaslist[2 * index - 4][j])
                    if (new_md2) < tempmin:
                        self.curScorelist[2 * index - 4] -= (tempmin - new_md2)

                    self.curScorelist[2 * index - 3] += abs(new_md2 - self.deltaslist[2 * index - 4][j]) - abs(
                        self.deltaslist[2 * index - 3][j] - self.deltaslist[2 * index - 4][j])
                    self.deltaslist[2 * index - 3][j] = new_md2

                    mid_delta_value = abs(self.deltaslist[2 * index - 3][j] - self.deltaslist[2 * index - 4][j])
                    new_mid_w = min(self.deltaslist[2 * index - 3][j], self.deltaslist[2 * index - 4][j]) - self.alpha * mid_delta_value
                    self.dtrees[index - 1].setVal(j, new_mid_w)
                else:
                    self.dtrees[index - 1].changeVal(j, -self.mtranslist[index - 1][mindelta, j])

            self.sets[index] -= {mindelta}
            self.dtrees[index].changeVal(mindelta, float('inf'))
            self.deleted.append((index, mindelta))
            self.numDeleted += 1
            
    
    # find the smallest one in all set
    def findmin(self):
        resmin = []
        for i in range(len(self.dtrees)):
            (next, delt) = self.dtrees[i].getMin()
            resmin.append((next, delt))

        min1 = resmin[0][1] * (1 + self.alpha) 
        min2 = resmin[-1][1] * (1 + self.alpha)
        min_weight = min1
        minidx = resmin[0][0]
        idx = 0
        mold = 0
        
        if min2 < min_weight:
            min_weight = min2
            idx = len(resmin) - 1
            mold = 2
            minidx = resmin[-1][0]
        for i in range(len(resmin) - 2):
            if resmin[i + 1][1] < min_weight:
                min_weight = resmin[i + 1][1]
                minidx = resmin[i + 1][0]
                idx = i + 1
                mold = 1
        
        return mold, idx, minidx
        
        
    def checkset(self):
        res = True
        for i in range(len(self.sets)):
            res = self.sets[i] and res
        return res

    
    # use to print the result of algorithm
    def printres(self):
        print('best delete number : ', self.bestNumDeleted)
        res1 = 'nodes number remaining:  '
        for i in range(len(self.sets)):
            res1 += str(len(self.sets[i])) + ' '
        print(res1)
        res2 = 'matrix mass remaining:  '
        for i in range(len(self.curScorelist)):
            res2 += str((self.curScorelist[i])) + ' '
        print(res2)

        print('best score of g(S): ', self.bestAveScore)
        res3 = 'min value of the tree :  '
        for i in range(len(self.dtrees)):
            res3 += str((self.dtrees[i].getMin())) + ' '
        print(res3)
    
    
    def fastGreedyDecreasing(self):
        print('this is the cpu version of FlowScope')  
        print('start  greedy')

        self.initGreedy()

        self.bestAveScore = float('-inf')

        self.numDeleted = 0
        self.deleted = []
        self.bestNumDeleted = 0

        curall = 0
        for i in range(0, len(self.curScorelist), 2):
            curall += self.curAveScorelist[i] - self.alpha * self.curAveScorelist[i+1]
        print('initial score of g(S):', curall)
        while self.checkset():  # repeat deleting until one node set in null

            mold, idx, minidx = self.findmin()

            self.updataConnNode(mold=mold, index=idx, mindelta=minidx)
            
            s = 0
            for i in range(len(self.sets)):
                s += len(self.sets[i])
            if (s) > 0:
                for i in range(0, len(self.curAveScorelist), 2):
                    self.curAveScorelist[i] = self.curScorelist[i] / s
                    self.curAveScorelist[i+1] = self.curScorelist[i+1] / s
            else:
                print("something wrong in FlowScope")
                for i in range(0, len(self.curAveScorelist), 2):
                    self.curAveScorelist[i] = 0
                    self.curAveScorelist[i+1] = 0 
                    
            curAveScore =0
            for i in range(0, len(self.curAveScorelist), 2):
                curAveScore += self.curAveScorelist[i] - self.alpha * self.curAveScorelist[i+1]

            
            if curAveScore >= self.bestAveScore: 
                self.bestNumDeleted = self.numDeleted
                self.bestAveScore = curAveScore

        self.printres()
        

        finalsets = []
        finalsets.append(set(range(self.mcurlist[0].shape[0])))
        for i in range(len(self.mcurlist) - 1):
            finalsets.append(set(range(self.mcurlist[i].shape[1])))
        finalsets.append(set(range(self.mcurlist[-1].shape[1])))
        
        for i in range(self.bestNumDeleted):
            finalsets[self.deleted[i][0]].remove(self.deleted[i][1])
            

        return finalsets, self.bestAveScore


    def anomaly_detection(self, k:int=3, alpha:float = 0.8):
        return self.run(k=k, alpha=alpha)
