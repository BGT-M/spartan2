
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


def loadedgelist2sm(edgelist, mtype=csc_matrix, dtype=int, delimiter=' ',
                idstartzero=True, issquared=False):
    '''
    load edge list into sparse matrix
    matrix dimensions are decided by max row id and max col id
    support csr, coo, csc matrix
    '''
    xs=[]
    ys=[]
    data=[]
    if idstartzero is True:
        offset = 0
    else:
        offset = -1
    for coords in edgelist:
        xs.append(int(coords[0]) + offset)
        ys.append(int(coords[1]) + offset)
        if len(coords) == 3:
            data.append(dtype(coords[2]))
        else:
            data.append(1)
    m = max(xs) + 1
    n = max(ys) + 1
    if issquared is False:
        M, N = m, n
    else:
        M = max(m,n)
        N = max(m,n)
    sm = mtype((data, (xs, ys)), shape=(M,N))
    return sm

def saveSimpleListData(simls, outdata):
    with open(outdata, 'w') as fw:
        fw.write('\n'.join(map(str, simls)))
        fw.write('\n')
        fw.close()
