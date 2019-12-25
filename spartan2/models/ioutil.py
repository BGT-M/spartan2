
import numpy as np
from scipy.sparse import coo_matrix


def loadedgelist2sm(edgelist, mtype=coo_matrix, dtype=int, delimiter=' ',
                    idstartzero=True, issquared=False):
    '''
    load edge list into sparse matrix
    matrix dimensions are decided by max row id and max col id
    support csr, coo, csc matrix
    '''
    data = []
    if idstartzero is True:
        offset = 0
    else:
        offset = -1

    coods = np.transpose(edgelist)
    print(coods.shape)
    xs, ys = (coods[0]+offset).astype(int), (coods[1]+offset).astype(int)
    if coods.shape[0] >= 3:
        "by default the last row store values"
        data = (coods[-1]).astype(dtype)
    else:
        print("Warning: no values in the input data.")
        data = np.ones(coods.shape[1], int)
    m = max(xs) + 1
    n = max(ys) + 1
    if issquared is False:
        M, N = m, n
    else:
        M = max(m, n)
        N = max(m, n)
    sm = mtype((data, (xs, ys)), shape=(M, N))
    return sm


def saveSimpleListData(simls, outdata):
    with open(outdata, 'w') as fw:
        fw.write('\n'.join(map(str, simls)))
        fw.write('\n')
        fw.close()
