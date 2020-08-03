from numpy import random

from .SinglePassPCA import generateGH_by_multiply, generateQB, computeSVD
from .util import calDensity, findSuspWins, filterEigenvec, getKeys
from spartan.tensor import TensorData, DTensor
from .._model import DMmodel
from spartan.util.basicutil import StringMapper
from spartan.util.drawutil import drawEigenPulse


class EigenPulse(DMmodel):
    def __init__(self, stream_tensor, **param_dict):
        self.stream_tensor = stream_tensor
        self.window = param_dict['window']
        self.stride = param_dict['stride']
        self.l = param_dict['l']
        self.b = param_dict['b']
        self.ts_colidx = param_dict['ts_idx']
        self.item_colidx = param_dict['item_idx']
        self.stringMapper = StringMapper()
        self.mappers = {self.item_colidx: self.stringMapper}

    def anomaly_detection(self):
        return self.run()

    def run(self):
        densities, submats = [], []
        winidx = 0
        while True:
            try:
                tensorlist = self.stream_tensor.fetch_slide_window(self.window, self.stride, self.ts_colidx)
                tensorlist[self.item_colidx] = \
                    tensorlist[[self.item_colidx, self.ts_colidx]].apply(tuple, axis=1)
                del tensorlist[self.ts_colidx]

                td = TensorData(tensorlist)
                stensor = td.toSTensor(self.stream_tensor.hasvalue, self.mappers)
 
                n = stensor.shape[1]
                
                Omg = DTensor.from_numpy(x=random.randn(n, self.l))
                G, H = generateGH_by_multiply(stensor, Omg)
                Q, B = generateQB(G, H, Omg, self.l, self.b)
                u1, s1, v1 = computeSVD(Q, B)
                u1 = u1.T
                submat, rowids, colids = filterEigenvec(stensor, u1, v1)
                submats.append((submat, rowids, colids))

                density = calDensity(submat)
                print(f"density of dense submatrix in window {winidx} is {density}")
                densities.append(density)
                winidx += 1
            except:
                break
        
        susp_wins = findSuspWins(densities)
        print('suspicious windows:', susp_wins)
        res = []
        for susp_win in susp_wins:
            submat, rowids, colids = submats[susp_win]
            items, tss = getKeys(colids, self.mappers[1].strdict)
            res.append((susp_win, rowids, items, tss, densities[susp_win]))
        for block in res:
            print('suspicious window:', block[0])
            print('user list of dense block:', block[1])
            print('item list of dense block:', block[2])
            print('time list of dense block:', block[3])
            print('density of dense block:', block[4])
        return res, densities
