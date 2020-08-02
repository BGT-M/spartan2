import numpy as np
import math


def filterEigenvec(A, u, v, k=0):
    m = A.shape[0]
    n = A.shape[1]
    num1 = 1 / math.sqrt(m)
    num2 = 1 / math.sqrt(n)
    list_u = u.tolist()
    list_v = v.tolist()
    list_i = [j for j, x in enumerate(list_u[k]) if abs(x) >= num1]
    list_j = [j for j, x in enumerate(list_v[k]) if abs(x) >= num2]
    B1 = A[list_i, :]
    B = B1[:, list_j]
    return B, list_i, list_j


def calDensity(mat):
    num = mat.sum()
    density = round(float(num) / (mat.shape[0] + mat.shape[1]), 2)
    return density


def findSuspWins(densities):
    mean = np.mean(densities)
    std = np.std(densities, ddof=1)
    thres = 3 * std + mean
    print('mean:{}, std val:{}, thres:{}'.format(mean, std, thres))
    burst_wins = []
    for i in range(densities.__len__()):
        if densities[i] > thres:
            burst_wins.append(i)
    return burst_wins


def getKeys(vs, dict):
    items, times = [], []
    for k in dict:
        v = dict[k]
        if v in vs:
            items.append(k[0])
            times.append(k[1])
    return items, times
        

