import numpy as np
from .._model import DMmodel
import scipy.sparse.linalg as slin
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from spartan.model.fraudar.greedy import logWeightedAveDegree

class Fraudar(DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, out_path = "./", file_name = "out"):
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        res = logWeightedAveDegree(sparse_matrix)
        print(res)
        np.savetxt("%s.rows" % (out_path + file_name, ), np.array(list(res[0][0])), fmt='%d')
        np.savetxt("%s.cols" % (out_path + file_name, ), np.array(list(res[0][1])), fmt='%d')
        print("score obtained is ", res[1])
    
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
