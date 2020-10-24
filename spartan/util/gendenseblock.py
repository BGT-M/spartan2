import sys
import numpy as np
import random
from .ioutil import saveSimpleListData

def generateProps(stensor, i, j, s, t0, tsdiffcands, tsp, inject_coords, inject_values):

    # rs = np.random.choice([4, 4.5], size=s)
    rs = np.random.choice([4, 5], size=s)
    ts = np.random.choice(tsdiffcands, size=s, p=tsp) + t0
    maxTime = stensor.shape[2]
    for k in range(len(rs)):
        while ts[k] > maxTime:
            ts[k] = np.random.choice(tsdiffcands, size=1, p=tsp) + t0
        inject_coords.append([i, j, ts[k], rs[k]])
        inject_values.append(1)
    return

def injectFraud2PropGraph(stensor, acnt, bcnt, goal, popbd,
                          testIdx = 3, idstartzero=True, re=True, suffix=None,
                         weighted=True, output=True, multiedge=False, dataset='yelp'):
    if not idstartzero:
        print('we do not handle id start 1 yet for ts and rate')
        # ratefile, tsfile = None, None
    'smax: the max # of multiedge'
    if multiedge:
        smax = stensor.max() # max freqency
    else:
        smax = 1

    if acnt == 0 and re:
        return tensorData, ([], [])
    (m, n, p, q) = stensor.shape
    rates, times, tsdiffs, t0 = [], [], [], 0
    t0, tsdiffcands,tsp = 0, [], []
    inject_coords, inject_values = [], []

    tsmin, tsmax = sys.maxsize,0
    tsdiffs = np.array([])
    prodts={i:[] for i in range(n)}

    nonzerocoords =  np.nonzero(stensor)
    nonzerocoords = np.asarray(np.nonzero(stensor))
    for i in range(nonzerocoords.shape[1]):
        pid = nonzerocoords[1][i]
        prodts[pid].append(nonzerocoords[2][i])

    for pv in prodts.values():
        pv = sorted(pv)
        minv, maxv = pv[0], pv[-1]
        if tsmin > minv:
            tsmin = minv
        if tsmax < maxv:
            tsmax = maxv
        if len(pv)<=2:
            continue
        vdiff = np.diff(pv)
        'concatenate with [] will change value to float'
        tsdiffs = np.concatenate((tsdiffs, vdiff[vdiff>0]))
    
    tsdiffs.sort()
    tsdiffs = tsdiffs.astype(int)
    tsdiffcands = np.unique(tsdiffs)[:20] #another choice is bincount
    tsp = np.arange(20,dtype=float)+1
    tsp = 1.0/tsp
    tsp = tsp/tsp.sum()
    t0 = np.random.randint(tsmin, tsmax, dtype=int)
    
    if dataset == 'yelp':
        colSum = stensor.sum(axis=3).sum(axis=2).sum(axis=0)
    elif dataset == 'wb':
        colSum = stensor.sum(axis=2).sum(axis=0) #need test
    colSum = colSum.todense()

    colids = np.arange(n, dtype=int)
    targetcands = np.argwhere(colSum < popbd).flatten()
    targets = random.sample(list(targetcands), bcnt)
    camocands = np.setdiff1d(colids, targets, assume_unique=True)
    camoprobs = colSum[camocands]/float(colSum[camocands].sum())
    #population = np.repeat(camocands, colSum[camocands].astype(int), axis=0)
    fraudcands = np.arange(m,dtype=int) #users can be hacked
    fraudsters = random.sample(list(fraudcands), acnt)
    'rating times for one user to one product, multiedge'
    scands = np.arange(1,smax+1,dtype=int)
    sprobs = []
    if multiedge:
        numedges = stensor.sum()
        for s in scands:
            nums = (stensor==s).sum()
            sprobs.append(float(nums)/numedges)
    else:
        sprobs.append(1)

    # inject near clique
    for j in targets:
        exeusers = random.sample(list(fraudsters), goal)
        for i in exeusers:
            s = np.random.choice(scands, size=1, p=sprobs)[0] if weighted else 1
            # if (not weighted) and M2[i,j] > 0:
            if (not weighted) and stensor[i][j].sum() > 0: 
                continue
            generateProps(stensor, i, j, s, t0, tsdiffcands, tsp, inject_coords, inject_values)

    # inject camo
    p = goal/float(acnt)
    for i in fraudsters:
        if testIdx == 1:
            thres = p * bcnt / (n - bcnt)
            for j in camocands:
                s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                if (not weighted) and stensor[i][j].sum() > 0:
                    continue
                if random.random() < thres:
                    generateProps(stensor, i, j, s, t0, tsdiffcands, tsp, inject_coords, inject_values)
        if testIdx == 2:
            thres = 2 * p * bcnt / (n - bcnt)
            for j in camocands:
                s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
                if (not weighted) and stensor[i][j].sum() > 0:
                    continue
                if random.random() < thres:
                    generateProps(stensor, i, j, s, t0, tsdiffcands, tsp, inject_coords, inject_values)
        # biased camo           
        if testIdx == 3:
            colRplmt = np.random.choice(camocands, size=int(bcnt*p),
                                        p=camoprobs)
            #M2[i,colRplmt] = 1
            s = np.random.choice(scands, size=1, p=sprobs) if weighted else 1
            for j in colRplmt:
                if (not weighted) and stensor[i][j].sum() > 0:
                    continue
                generateProps(stensor, i, j, s, t0, tsdiffcands, tsp, inject_coords, inject_values)

    inject_coords = np.asarray(inject_coords)
    inject_values = np.asarray(inject_values)
    if suffix is not None:
        suffix = str(suffix)
    else:
        suffix =''
    if output is True:
        injtrueA = dataset+'.trueA'+suffix
        injtrueB = dataset+'.trueB'+suffix
        saveSimpleListData(fraudsters, injtrueA)
        saveSimpleListData(targets, injtrueB)
    return (inject_coords, inject_values), (fraudsters, targets)