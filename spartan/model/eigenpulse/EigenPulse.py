from numpy import random

from .SinglePassPCA import generateGH_by_multiply, generateQB, computeSVD
from .util import calDensity, findSuspWins, filterEigenvec
from spartan.tensor import TensorData, DTensor
from .._model import DMmodel
from spartan.util.basicutil import StringMapper


class EigenPulse(DMmodel):
    def __init__(self, stream_tensor, **param_dict):
        self.stream_tensor = stream_tensor
        self.window = param_dict['window']
        self.stride = param_dict['stride']
        self.l = param_dict['l']
        self.b = param_dict['b']
        self.ts_colidx = param_dict['ts_idx']
        self.item_colidx = param_dict['item_idx']
        self.hasvalue = param_dict['hasvalue']
        self.mappers = {self.item_colidx: StringMapper()}
    
    def anomaly_detection(self):
        return self.run()
    
    def run(self):
        densities, submats = [], []
        winidx = 0
        while True:
            try:
                tensorlist = self.stream_tensor.fetch_slide_window(self.window, self.stride, self.ts_colidx)
                tensorlist[self.item_colidx] = tensorlist[self.item_colidx].map(str) \
                                               + tensorlist[self.ts_colidx].map(str)
                del tensorlist[self.ts_colidx]
                
                td = TensorData(tensorlist)
                stensor = td.toSTensor(self.hasvalue, self.mappers)
                n = stensor.shape[1]
                Omg = DTensor.from_numpy(x=random.randn(n, self.l))
        
                G, H = generateGH_by_multiply(stensor, Omg)
                Q, B = generateQB(G, H, Omg, self.l, self.b)
                u1, s1, v1 = computeSVD(Q, B)
                u1 = u1.T
                submat = filterEigenvec(stensor, u1, v1)
                submats.append(submat)
        
                density = calDensity(submat)
                print(f"density of dense submatrix in window {winidx} is {density}")
                densities.append(density)
                winidx += 1
            except:
                break
        print('densities of detected block in all windows:', densities)
        susp_wins = findSuspWins(densities)
        print('suspicious windows:', susp_wins)
        susp_submats = []
        for susp_win in susp_wins:
            susp_submats.append(submats[susp_win])
        return susp_wins, susp_submats

    









