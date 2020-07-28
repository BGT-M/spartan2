import numpy as np
from scipy import sparse
from scipy.linalg import svd
import math
from spartan.tensor import DTensor


def generateGH_by_multiply(A, Omg):
    G = A.dot(Omg)
    H = A.T.dot(G)
    return G, H


def generateGH_by_list(G, H, glist, hlist, k):
    if k == 0:
        for g in glist:
            G = np.vstack((G, g))
        for h in hlist:
            H = H + h
    else:
        g = glist.pop(0)
        gm = g.shape[0]
        G = G[gm:, :]
        h = hlist.pop(0)
        H = H - h
        G = np.vstack((G, glist[-1]))
        H = H + hlist[-1]
    return G, H, glist, hlist


def generateQB(G, H, Omg, l, b):
    m = G.shape[0]
    n = H.shape[0]
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))
    t = int(math.floor(l/b))
    for i in range(0, t):
        if B.shape[0] == 0:
            Yi = G[:, i*b: (i+1)*b]
            Qi, Ri = np.linalg.qr(Yi)
            Qi, Rit = np.linalg.qr(Qi)
            Ri = Rit * Ri
            invRi = DTensor.from_numpy(x=np.linalg.inv(Ri.T))
            Bi = invRi.dot(H[:, i * b: (i + 1) * b].T)
        else:
            temp = B.dot(Omg[:, i*b: (i+1)*b])
            Yi = G[:, i*b: (i+1)*b] - Q.dot(temp)
            Qi, Ri = np.linalg.qr(Yi)
            Qi, Rit = np.linalg.qr(Qi - Q.dot(Q.T.dot(Qi)))
            Ri = Rit * Ri
            invRi = DTensor.from_numpy(x=np.linalg.inv(Ri.T))
            dt_B, dt_Q = DTensor.from_numpy(B), DTensor.from_numpy(Q)
            Bi = invRi.dot(H[:, i*b: (i+1)*b].T - Yi.T.dot(dt_Q).dot(dt_B) - temp.T.dot(dt_B))
        Q = np.hstack((Q, Qi))
        B = np.vstack((B, Bi))
    return Q, B


def computeSVD(Q, B):
    u1, s, v = svd(B, full_matrices=False)
    u = Q.dot(u1)
    return u, s, v
