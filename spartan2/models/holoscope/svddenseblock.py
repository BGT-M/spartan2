import sys
sys.path.append('../')
from gendenseblock import *
from mytools.ioutil import saveSimpleListData
import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csr_matrix
from gendenseblock import *
from os.path import expanduser
import math
home = expanduser("~")

def svddenseblock(m, rbd='avg'):
    m = m.asfptype()
    u, s, vt = slin.svds(m, k=1, which='LM')
    u1 = u[:,0]
    v1 = vt[0,:]
    s1 = s[0]
    if abs(max(u1)) < abs(min(u1)):
        u1 = -1*u1
    if abs(max(v1)) < abs(min(v1)):
        v1 = -1*v1
    sqrtS1 = math.sqrt(s1)
    if type(rbd) is float:
       rows = ((sqrtS1*u1)>=rbd).astype(int)
       cols = ((sqrtS1*v1)>=rbd).astype(int)
    elif rbd == 'avg':
       nrow, ncol = m.shape
       rows = (u1>=1.0/math.sqrt(nrow)).astype(int)
       cols = (v1>=1.0/math.sqrt(ncol)).astype(int)
    #rows = np.round(sqrtS1*u1).astype(int)
    #cols = np.round(sqrtS1*v1).astype(int)
    return rows, cols

def svddenseblockrank(m):
    m = m.asfptype()
    u, s, vt = slin.svds(m, k=1, which='LM')
    u1 = u[:,0]
    v1 = vt[0,:]
    s1 = s[0]
    if abs(max(u1)) < abs(min(u1)):
        u1 = -1*u1
    if abs(max(v1)) < abs(min(v1)):
        v1 = -1*v1
    sqrtS1 = math.sqrt(s1)
    rows = sqrtS1*u1
    cols = sqrtS1*v1
    return rows, cols

if __name__=="__main__":
    datapath=home+'/Data/'
    testdatapath='./testdata/'
    respath='./testout/'
    dataname='example.txt'
    data=testdatapath+dataname
    #coom = readedge2coom(data, weighted=False, idstartzero=True)
    print('loading data ... ...')

    '''
    M = genTriDenseBlock(1000, 1000, 1000, 500, 1000,1000, p1=0.8, alpha2=3,
                         alpha3=9.0)
    sm=csr_matrix(M)
    '''
    '''
    m = genDiDenseBlock(500,500, 1, 1500, 1500, alpha=-1)
    m=addnosie(m, 2000, 2000, 0.4, black=False)
    m=addnosie(m, 2000, 500, 0.005, black=True, A0=500, B0=0)
    m=addnosie(m, 500, 2000, 0.005, black=True, A0=0, B0=500)
    '''
    #m = genTriRectBlocks(3000,3000,0.6,0.6,0.6)
    #m = genDiHyperRectBlocks(50, 50, 2500, 2500, alpha=-0.5)
    '''when hiperbola: contains 50x50 core, rect: 88x88 can be
    detected by genDiHyperRectBlocks (optimal)
       rect: < 88x88 will not be detected, instead it detects the 58x58 in
       hyperbolia
    '''
    A1,B1,A2,B2= 500,500, 2500, 2500 #100,100, 2500, 2500 #88, 88, 2500, 2500,
    m = genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=0.002)
    m = addnosie(m, A1+A2, B1+B2, 0.005, black=True, A0=0, B0=0)
    m = addnosie(m, A1+A2, B1+B2, 0.4, black=False, A0=0, B0=0)
    sm=csr_matrix(m, dtype=np.float64)
    drows, dcols = svddenseblock(sm)
    print("dense block size: {}x{}".format(len(drows.nonzero()[0]),
                                           len(dcols.nonzero()[0])))

