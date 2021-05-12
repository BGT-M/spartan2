import numpy as np
from .._model import DMmodel
import scipy.sparse.linalg as slin
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from spartan.model.fraudar.greedy import logWeightedAveDegree

class Fraudar(DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, out_path = "./", file_name = "out", k = 1, max_size = -1):
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        Mcur = sparse_matrix.copy().tolil()
        res = []
 
        while (t < k):
            list_row, list_col, score = logWeightedAveDegree(Mcur)
            print("Fraudar iter %s finished." % t)
            
            if isinstance(max_size, int):
                if (max_size==-1 or (max_size>=len(list_row) and max_size>=len(list_col))):
                    t += 1
                    res.append((list_row, list_col, score))
            elif max_size[0]>=len(list_row) and max_size[1]>=len(list_col):
                t += 1
                res.append((list_row, list_col, score))
            if (t >= k) break

            (rs, cs) = Mcur.nonzero() # (u, v)
            ## only delete inner connections
            rowSet = set(list_row)
            colSet = set(list_col)
            for i in range(len(rs)):
                if rs[i] in rowSet and cs[i] in colSet:
                    Mcur[rs[i], cs[i]] = 0

            np.savetxt("%s_%s.rows" % (out_path + file_name, t), np.array(list_row), fmt='%d')
            np.savetxt("%s_%s.cols" % (out_path + file_name, t), np.array(list_col), fmt='%d')
            print("score obtained is ", res[1])
    
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
