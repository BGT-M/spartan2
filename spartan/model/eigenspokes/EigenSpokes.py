import numpy as np
from .._model import DMmodel
import scipy.sparse.linalg as slin
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class EigenSpokes(DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, k = 10):
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        RU, RS, RVt = slin.svds(sparse_matrix, k)
        RV = np.transpose(RVt)
        U, S, V = np.flip(RU, axis=1), np.flip(RS), np.flip(RV, axis=1)

        n_row = U.shape[0]
        n_col = V.shape[0]

        x_lower_bound = -1 / np.sqrt(n_col + 1)
        y_lower_bound = -1 / np.sqrt(n_col + 1)
        x_upper_bound = 1 / np.sqrt(n_col + 1)
        y_upper_bound = 1 / np.sqrt(n_col + 1)

        real_index1 = S.shape[0] - 1
        real_index2 = S.shape[0] - 2

        x = U[:, real_index1]
        y = U[:, real_index2]
    

        list_x_lower_outliers = [index for index in range(len(x)) if x[index] > x_lower_bound]
        list_y_lower_outliers = [index for index in range(len(y)) if y[index] > y_lower_bound]
        list_x_upper_outliers = [index for index in range(len(x)) if x[index] < x_upper_bound]
        list_y_upper_outliers = [index for index in range(len(y)) if y[index] < y_upper_bound]

        outliers_index = list(set(list_x_lower_outliers) & set(list_y_lower_outliers) &
                              set(list_x_upper_outliers) & set(list_y_upper_outliers))
        inliers_index = list(set(range(len(x))).difference(outliers_index))

        outliers_index, inliers_index = inliers_index, outliers_index
        print("Outliters:")
        print(outliers_index)
        return outliers_index
    
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
