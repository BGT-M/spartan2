import sys,math
import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from svddenseblock import *
from mytools.ioutil import myreadfile
from os.path import expanduser
home = expanduser("~")


def loadtensor2matricization(tensorfile, sumout=[], mtype=coo_matrix,
                             weighted=True, dtype=int):
    'sumout: marginized (sumout) the given ways'
    matcols={}
    rindexcols={}
    xs, ys, data = [], [], []
    with myreadfile(tensorfile, 'rb') as f:
        for line in f:
            elems = line.strip().split(',')
            elems = np.array(elems)
            u = int(elems[0])
            colidx = range(1,len(elems)-1) #remove sumout
            colidx = set(colidx) - set(list(sumout))
            colidx = sorted(list(colidx))
            col=' '.join(elems[colidx])
            if col not in matcols:
                idx = len(matcols)
                matcols[col] = idx
                rindexcols[idx]=col
            cid = matcols[col]
            w = dtype(elems[-1])
            xs.append(u)
            ys.append(cid)
            data.append(w)
        nrow, ncol = max(xs)+1, max(ys)+1
        sm = mtype( (data, (xs, ys)), shape=(nrow, ncol), dtype=dtype )
        if weighted is False:
            sm.data[0:] = dtype(1)
        f.close()

    return sm, rindexcols

def matricizeSVDdenseblock(sm, rindexcols, rbd='avg'):
    A, tmpB = svddenseblock(sm, rbd=rbd)
    rows = A.nonzero()[0]
    cols = tmpB.nonzero()[0]
    bcols = set()
    for col in cols:
        'col name'
        cnm = rindexcols[col]
        cnm = cnm.strip().split(' ')
        b = int(cnm[0])
        bcols.add(b)
    return set(rows), set(bcols)

if __name__=="__main__":
    path = home+'/Data/BeerAdvocate/'
    respath= path+'results/'
    tsfile = path+'userbeerts.dict'
    ratefile = path+'userbeerrate.dict'
    tensorfile =respath+'userbeer.tensor'
    sm, rindexcols = loadtensor2matricization(tensorfile,
                                              sumout=[3],mtype=csr_matrix,
                                              dtype=float,weighted=True)
    A, B = matricizeSVDdenseblock(sm, rindexcols, rbd='avg')

