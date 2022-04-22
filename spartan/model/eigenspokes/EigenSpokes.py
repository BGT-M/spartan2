import numpy as np
from .._model import DMmodel
import scipy.sparse.linalg as slin
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class EigenSpokes(DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, k = 0, is_directed = False):
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        RU, RS, RVt = slin.svds(sparse_matrix, k+3)
        RV = np.transpose(RVt)
        U, S, V = np.flip(RU, axis=1), np.flip(RS), np.flip(RV, axis=1)

        m = U.shape[0]
        n = V.shape[0]

        x = U[k] * -1 if np.abs(np.min(U[k])) > np.abs(np.max(U[k])) else U[k]
        y = V[k] * -1 if np.abs(np.min(V[k])) > np.abs(np.max(V[k])) else V[k]

        x_outliers = [index for index in range(len(x)) if x[index] > 1 / np.sqrt(m)]
        y_outliers = [index for index in range(len(y)) if y[index] > 1 / np.sqrt(n)]
 
        if is_directed:
            return (x_outliers, y_outliers)
        else:
            return x_outliers
    
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
