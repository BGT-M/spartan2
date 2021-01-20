import sys, os, time
import numpy as np
import scipy as sci
import scipy.stats as ss
import scipy.sparse.linalg as slin
import copy
from .mytools.MinTree import MinTree
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from .mytools.ioutil import loadedge2sm
from .edgepropertyAnalysis import MultiEedgePropBiGraph
import math
from .._model import DMmodel
from spartan.util.basicutil import param_default
from spartan.backend import STensor

def score_level_objects( objscores, p=0.90):
    '''implement with Perato distribution, given significant value
    '''
    sortscores = np.array(sorted(objscores))
    sortobjs = np.argsort(objscores)
    alpha = 0.9
    tail_fir_score = np.percentile(sortscores, [alpha*100])[0]
    if tail_fir_score == 0:
        'remove 0 if the number of percentile 90% is 0'
        firindex = np.argwhere(sortscores > 0)[0]
        sortscores = sortscores[firindex[0]:]
        sortobjs = sortobjs[firindex[0]:]
    'fit generalized pareto distribution using 10% upper tail data'
    tailidx = int(alpha * len(sortscores))
    tailscores = sortscores[tailidx:]
    tailobjs = sortobjs[tailidx:]

    shape, pos, scale = ss.pareto.fit(tailscores)
    cdfs = ss.pareto.cdf(tailscores, shape, pos, scale)

    levelidxs = np.argwhere(cdfs >= p)
    levelobjs = tailobjs[levelidxs].T[0]
    return levelobjs

def score_heristic_level_objects( objscores ):
    '''todo: implement with Perato distribution, given significant value
    '''
    sortscores = sorted(objscores, reverse=True)
    sortobjs = np.argsort(objscores)[::-1]
    diffscores = - np.diff(sortscores)
    levelid = np.argmax(diffscores)
    levelobjs = sortobjs[ : levelid+1]
    return levelobjs

def nonzero_objects( objscores ):
    objects = np.where( objscores > 0 )[0]
    return objects

class Ptype(object):
    freq =0
    ts = 1
    rate=2
    @staticmethod
    def ptype2str(p):
        if p == Ptype.freq:
            return 'freq'
        if p == Ptype.ts:
            return 'ts'
        if p == Ptype.rate:
            return 'rate'
    @staticmethod
    def ptypes2str(ptypes):
        strs=[]
        if Ptype.freq in ptypes:
            strs.append(Ptype.ptype2str(Ptype.freq))
        if Ptype.ts in ptypes:
            strs.append(Ptype.ptype2str(Ptype.ts))
        if Ptype.rate in ptypes:
            strs.append(Ptype.ptype2str(Ptype.rate))
        pstr = '-'.join(strs)
        return pstr

class HoloScopeOpt:
    def __init__(self, graphmat, qfun='exp', b=32,
                 aggmethod='sum', sdrop=True, mbd=0.5, sdropscale='linear',
                 tsprop=None, tunit='s', rateprop=None):
        'how many times of a user rates costumers if he get the cost balance'
        self.coe = 0
        'the larger expbase can give a heavy penalty to the power-law curve'
        self.expbase = b
        self.scale = qfun
        self.b = b
        self.aggmethod=aggmethod
        self.suspbd = 0.0 #susp < suspbd will assign to zero
        self.priordropslop=sdrop

        self.graph=graphmat.tocoo()
        self.graphr = self.graph.tocsr()
        self.graphc = self.graph.tocsc()
        self.matricizetenor=None
        self.nU, self.nV=graphmat.shape
        self.indegrees = graphmat.sum(0).getA1()
        self.e0 = math.log(graphmat.sum(), self.nU) #logrithm of edges
        print('matrix size: {} x {}\t#edges: {}'.format(self.nU, self.nV,
                                                          self.indegrees.sum()))

        # tunit is only used for files input
        self.tsprop, self.rateprop, self.tunit = tsprop, rateprop, tunit
        self.tspim, self.ratepim = None, None

        'field for multiple property graph'
        if tsprop is not None or rateprop is not None:
            if self.priordropslop:
                self.orggraph = self.graphr.copy()
            else:
                self.orggraph = self.graphr
        if tsprop is not None:
            self.mbd = mbd #multiburst bound
            self.tspim = MultiEedgePropBiGraph(self.orggraph)
            """
            since the data is cut by the end of time, so we need to see
            whether there is enough time twait from end of retweet to end of the
            whole data to judge if it is a sudden drop or cut by the end of time.
            twaits:
            """
            if isinstance(tsprop, str) and os.path.isfile(tsprop):
                self.tspim.load_from_edgeproperty(tsprop, mtype=coo_matrix,
                        dtype=np.int64)
                twaits = {'s':12*3600, 'h':24, 'd':30, None:0}
                twait = twaits[tunit]
            elif isinstance(tsprop, STensor):
                self.tspim.trans_array_to_edgeproperty(tsprop,
                        mtype=coo_matrix, dtype=np.int64)
                twait = 12
            else:
                raise Exception('Error: incorrect time stamp property')
            self.tspim.setup_ts4all_sinks(twait)
            if self.priordropslop:
                'slops weighted with max burst value'
                self.weightWithDropslop(weighted=True, scale=sdropscale)
        else:
            self.priordropslop = False #no input of time attribute
        if rateprop is not None:
            self.ratepim = MultiEedgePropBiGraph(self.orggraph)
            if isinstance(rateprop, str) and os.path.isfile(rateprop):
                self.ratepim.load_from_edgeproperty(rateprop, mtype=coo_matrix, dtype=float)
            elif isinstance(rateprop, STensor):
                self.ratepim.trans_array_to_edgeproperty(rateprop,
                        mtype=coo_matrix, dtype=float)
            else:
                raise Exception('Error: incorrect rate property')
            self.ratepim.setup_rate4all_sinks()

        'weighed with idf prior from Fraudar'
        #self.weightWithIDFprior()
        'if weighted the matrix the windegrees is not equal to indegrees'
        self.windegrees = self.graphc.sum(0).getA1()
        self.woutdegrees = self.graphr.sum(1).getA1()

        self.A = np.array([]) #binary array
        self.fbs = np.zeros(graphmat.shape[1], dtype=np.int) #frequency of bs in B
        '\frac_{ f_A{(bi)} }{ f_U{(bi)}}'
        self.bsusps = np.array([]) # the suspicious scores of products given A
        self.vx = 0 # current objective value
        self.vxs = [] #record all the vxs of optimizing iterations
        self.Y= np.array([])
        self.yfbs = np.array([])
        self.ybsusps = np.array([])
        'current is the best'
        self.bestvx = self.vx
        self.bestA = np.array([])
        self.bestfbs = np.array([])
        self.bestbsusps = np.array([])


    def weightWithDropslop(self, weighted, scale):
        'weight the adjacency matrix with the sudden drop of ts for each col'
        if weighted:
            colWeights = np.multiply(self.tspim.dropslops, self.tspim.dropfalls)
        else:
            colWeights = self.tspim.dropslops
        if scale == 'logistic':
            from scipy.stats import logistic
            from sklearn import preprocessing
            'zero mean scale'
            colWeights = preprocessing.scale(colWeights)
            colWeights = logistic.cdf(colWeights)
        elif scale == 'linear':
            from sklearn import preprocessing
            #add a base of suspecious for each edge
            colWeights = preprocessing.minmax_scale(colWeights) +1
        elif scale == 'plusone':
            colWeights += 1
        elif scale == 'log1p':
            colWeights = np.log1p(colWeights) + 1
        else:
            print('[Warning] no scale for the prior weight')

        n = self.nV
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.graphr = self.graphr * colDiag.tocsr()
        self.graph = self.graphr.tocoo(copy=False)
        self.graphc = self.graph.tocsc(copy=False)
        print("finished computing weight matrix")

    def weightWithIDFprior(self):
        print('weightd with IDF prior')
        colWeights = 1.0/np.log(self.indegrees + 5)
        n = self.nV
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.graphr = self.graphr * colDiag.tocsr()
        self.graph = self.graphr.tocoo(copy=False)
        self.graphc = self.graph.tocsc(copy=False)
        return

    'new objective with no f_A(v)/|A|'
    def maxobjfunc(self, A, fbs, bsusps=None):
        nu = 0.0
        de = 0.0
        numA = np.sum(A)
        de = numA + bsusps.sum() #math.sqrt(numA*bsusps.sum())#similar
        if numA == 0:
            return 0
        if bsusps is not None:
            nu = np.dot(fbs, bsusps)
        else:
            nu = fbs.sum()
        res = nu/np.float64( de )
        return res

    def aggregationMultiProp(self, mbs, method='sum'):
        if method == 'rank':
            from scipy.stats import rankdata
        rankmethod = 'average'
        k=60 #for rank fusion
        values = list(mbs.values())
        if len(mbs) == 1:
            val = values[0]
            if method == 'rank':
                rb = rankdata(-np.array(val), method=rankmethod)
                return np.reciprocal(rb+k) * k
            else:
                return val
        if method == 'sum':
            'this is the joint probability of exp form of prob'
            bsusps = values[0]
            for v in values[1:]:
                bsusps += v
        elif method == 'rank':
            'rank fusion'
            arrbsusps = []
            for val in values:
                rb = rankdata(-np.array(val), method=rankmethod)
                arrbsusps.append(np.reciprocal(rb+k))
            bsusps = np.array(arrbsusps).sum(0) * k
        else:
            print('[Error] Invalid method {}\n'.format(method))
        return bsusps

    #@profile
    def evalsusp4ts(self, suspusers, multiburstbd = 0.5, weighted=True):
        'the id of suspusers consistently starts from 0 no matter the source'
        incnt, inratio = self.tspim.suspburstinvolv(multiburstbd, weighted,
                                                    delta=True)
        suspts=inratio
        return suspts

    #@profile
    def evalsusp4rate(self, suspusers, neutral=False, scale='max'):
        susprates = self.ratepim.suspratedivergence(neutral, delta=True)
        if scale == 'max':
            if self.ratepim.maxratediv > 0:
                nsusprates = susprates/self.ratepim.maxratediv
            else:
                nsusprates = susprates
        elif scale=='minmax':
            #need a copy, and do not change susprates' value for delta
            from sklearn import preprocessing
            nsusprates = preprocessing.minmax_scale(susprates, copy=True)
        else:
            #no scale
            nsusprates = susprates
        return nsusprates

    'sink suspicious with qfunc, no f_A(v)/|A|'
    def prodsuspicious(self, fbs, A=None, scale='exp', ptype=[Ptype.freq]):
        multibsusps={}
        if Ptype.freq in ptype:
            posids = self.windegrees>0
            bs = np.zeros(self.nV)
            bs[posids] = np.divide(fbs[posids], self.windegrees[posids].astype(np.float64))
            multibsusps[Ptype.freq] = bs
        if Ptype.ts in ptype:
            suspusers = A.nonzero()[0]
            bs = self.evalsusp4ts(suspusers, multiburstbd=self.mbd)
            multibsusps[Ptype.ts] = bs
        if Ptype.rate in ptype:
            suspusers = A.nonzero()[0]
            bs = self.evalsusp4rate(suspusers)
            multibsusps[Ptype.rate] = bs
        bsusps = self.aggregationMultiProp(multibsusps, self.aggmethod)
        bsusps = self.qfunc(bsusps, fbs=fbs, scale=scale,
                numratios=len(multibsusps))
        return bsusps

    def initpimsuspects(self, suspusers, ptype):
        if Ptype.ts in ptype:
            self.tspim.setupsuspects(suspusers)
            temp1, temp2 = self.tspim.suspburstinvolv(multiburstbd=0.5, weighted=True,
                                       delta=False)
        if Ptype.rate in ptype:
            self.ratepim.setupsuspects(suspusers)
            tmp = self.ratepim.suspratedivergence(neutral=False,
                                            delta=False)
        return

    def start(self, A0, ptype=[Ptype.ts]):
        self.A = A0
        users = A0.nonzero()[0]
        self.ptype=ptype # the property type that the postiorer uses
        self.fbs = self.graphr[users].sum(0).getA1()
        self.fbs = self.fbs.astype(np.float64, copy=False)
        'initially set up currrent suspects'
        self.initpimsuspects(users, ptype=ptype)
        self.bsusps = self.prodsuspicious(self.fbs, self.A, ptype=ptype)
        self.vx = self.maxobjfunc(self.A, self.fbs, self.bsusps)
        self.vxs.append(self.vx)
        "current is the best"
        self.bestA = np.array(self.A)
        self.bestvx = self.vx
        self.bestfbs = np.array(self.fbs)
        self.bestbsusps = np.array(self.bsusps)

    def candidatefbs(self, z):
        'increase or decrease'
        coef = 1 if self.A[z] == 0 else -1
        bz = self.graphr[z]
        candfbs = (coef*bz + self.fbs).getA1()
        return candfbs

    #@profile
    def greedyshaving(self):
        '''greedy algorithm'''
        maxint = np.iinfo(np.int64).max//2
        delscores = np.array([maxint]*self.nU)
        delcands = self.A.nonzero()[0]
        deluserCredit = self.graphr[delcands,:].dot(self.bsusps)
        delscores[delcands] = deluserCredit
        print('set up the greedy min tree')
        MT = MinTree(delscores)
        i=0
        sizeA = np.sum(self.A)
        sizeA0 = sizeA
        setA = set(self.A.nonzero()[0])
        while len(setA) > 0:
            z, nextdelta = MT.getMin()
            setY = setA - {z}
            Y = copy.copy(self.A) # A is X
            Y[z] = 1-Y[z]
            self.Y=Y
            self.yfbs = self.candidatefbs(z)
            Ylist = Y.nonzero()[0]
            self.setdeltapimsusp(z, Ylist, add=False)
            self.ybsusps = self.prodsuspicious(self.yfbs, self.Y,
                                               ptype=self.ptype)
            vy = self.maxobjfunc(self.Y, self.yfbs, self.ybsusps)
            'chose next if next if the best'
            if vy > self.bestvx:
                self.bestA = np.array(self.Y)
                self.bestfbs = self.yfbs
                self.bestbsusps = self.ybsusps
                self.bestvx = vy
            MT.changeVal(z, maxint) #make the min to the largest for deletion
            prodchange = self.ybsusps - self.bsusps
            effectprod = prodchange.nonzero()[0]
            if len(effectprod)>0:
                #this is delta for all users
                userdelta = self.graphc[:,effectprod].dot(prodchange[effectprod])
                yuserdelta = userdelta[Ylist]
                for u in yuserdelta.nonzero()[0]:
                    uidx = Ylist[u]
                    MT.changeVal(uidx,yuserdelta[u])
            'delete next user, make current to next'
            self.A = self.Y
            sizeA -= 1
            setA = setY
            self.fbs = self.yfbs
            self.bsusps = self.ybsusps
            self.vx = vy
            self.vxs.append(self.vx)
            if i % (sizeA0//100 + 1) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            i+=1
        print()
        return np.sum(self.A)

    def initfastgreedy(self, ptype, numSing, rbd='avg', eps=1.6):
        '''
        default: ptype=[Ptype.freq], numSing=10, rbd='avg'
        '''
        self.ptype=ptype
        self.numSing=numSing #number of singular vectors we consider
        self.avgexponents=[]
        if len(ptype)==1:
            self.initfastgreedy2D(numSing, rbd, eps=eps)
        elif len(ptype) > 1:
            self.initfastgreedyMD(numSing, rbd, eps=eps)

        self.bestvx = -1
        self.qchop=False
        #reciprocal of indegrees
        self.sindegreciprocal = csr_matrix(self.windegrees).astype(np.float64)
        data = self.sindegreciprocal.data
        nozidx = data.nonzero()[0]
        self.sindegreciprocal.data[nozidx] = data[nozidx]**(-1)

        return

    def tenormatricization(self, tspim, ratepim, tbindic, rbins,
                           mtype=coo_matrix, dropweight=True, logdegree=False):
        'matricize the pim of ts and rates into matrix'
        if tspim is None and ratepim is None:
            return self.graph, range(self.nV)
        tscm, rtcm, dl = None, None,0
        if Ptype.ts in self.ptype and tspim is not None:
            tscm = tspim.edgeidxm.tocoo()
            dl = len(tscm.data)
        if Ptype.rate in self.ptype and ratepim is not None:
            rtcm = ratepim.edgeidxm.tocoo()
            dl = len(rtcm.data)
        if dropweight is True and tspim is not None:
            w = np.multiply(tspim.dropfalls, tspim.dropslops)
            w = np.log1p(w) + 1
        else:
            w = np.ones(self.nV)
        xs, ys, data, colWeights = [],[],[],[] # for matricized tenor
        matcols, rindexcols={},{}
        for i in range(dl):
            if tscm is not None and rtcm is not None:
                assert(tscm.row[i] == rtcm.row[i] and tscm.col[i] == rtcm.col[i])
                u = tscm.row[i]
                v = tscm.col[i]
                for t1, r1 in zip(tspim.eprop[tscm.data[i]],
                                ratepim.eprop[rtcm.data[i]]):
                    t = t1//int(tbindic[self.tunit])
                    r = rbins(r1)
                    strcol = ' '.join(map(str,[v,t,r]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            elif tscm is not None:
                u = tscm.row[i]
                v = tscm.col[i]
                for t1 in tspim.eprop[tscm.data[i]]:
                    t = t1//int(tbindic[self.tunit])
                    strcol = ' '.join(map(str,[v,t]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            elif rtcm is not None:
                u = rtcm.row[i]
                v = rtcm.col[i]
                for r1 in ratepim.eprop[rtcm.data[i]]:
                    r = rbins(r1)
                    strcol = ' '.join(map(str,[v,r]))
                    if strcol not in matcols:
                        idx = len(matcols)
                        matcols[strcol] = idx
                        rindexcols[idx]=strcol
                    xs.append(u)
                    ys.append(matcols[strcol])
                    data.append(1.0)
            else:
                print('Warning: no ts and rate for matricization')
                return self.graph, range(self.nV)

        nrow, ncol = max(xs)+1, max(ys)+1
        sm = mtype( (data, (xs, ys)), shape=(nrow, ncol), dtype=np.float64 )
        if logdegree:
            print('using log degree')
            sm.data[0:] = np.log1p(sm.data)
        if dropweight:
            m1, n1 = sm.shape
            for i in range(n1):
                pos = rindexcols[i].find(' ')
                v = int(rindexcols[i][:pos])
                colWeights.append(w[v])
            colDiag = lil_matrix((n1, n1))
            colDiag.setdiag(colWeights)
            sm = sm * colDiag.tocsr()
        return sm, rindexcols

    def initfastgreedyMD(self, numSing, rbd, eps = 1.6):
        '''
            use matricizationSVD instead of freq matrix svd
        '''
        #afile = self.tsprop if self.tsprop is not None else self.rateprop
        #ipath =  os.path.dirname(os.path.abspath(afile))
        tbindic={}
        if isinstance(self.tsprop, str) and os.path.isfile(self.tsprop):
            tbindic={'s':24*3600, 'd':30}
            print('Generate tensorfile with tunit:{}, tbins:{}'.format(self.tunit,
                                                                   tbindic[self.tunit]))
        elif isinstance(self.tsprop, STensor):
            tbindic={'s':1, 'd':1}
            print('Generate tensorfile with time rescale: ', tbindic[self.tunit] )

        'edgepropertyAnalysis has already digitized the ratings'
        rbins = lambda x: int(x) #lambda x: 0 if x<2.5 else 1 if x<=3.5 else 2
        if self.matricizetenor is None:
            matricize_start = time.clock()
            sm, rindexcol = self.tenormatricization(self.tspim, self.ratepim,
                    tbindic, rbins, mtype=coo_matrix,
                    dropweight=self.priordropslop,
                    logdegree=False)
            self.matricizetenor = sm
            print('::::matricize time cost: ', time.clock() - matricize_start)
        sm = self.matricizetenor
        print("matricize {}x{} and svd dense... ..."\
                .format(sm.shape[0], sm.shape[1]))
        u, s, vt = slin.svds(sm, k=numSing, which='LM')
        u = np.fliplr(u)
        s = s[::-1]
        CU, CV = [],[]
        for i in range(self.numSing):
            ui = u[:, i]
            si = s[i]
            if abs(max(ui)) < abs(min(ui)):
                ui = -1*ui
            if type(rbd) is float:
                sqrtSi = math.sqrt(si)
                ui *= sqrtSi
                rbdrow= rbd
            elif rbd == 'avg':
                rbdrow = 1.0/math.sqrt(self.nU)
            else:
                print('unkown rbd {}'.format(rbd))
            rows = np.argsort(-ui, axis=None, kind='quicksort')
            for jr in range(len(rows)):
                r = rows[jr]
                if ui[r] <= rbdrow:
                    break
            self.avgexponents.append(math.log(jr, self.nU))
            'consider the # limit'
            if self.nU > 1e6:
                e0 = self.e0
                ep = max(eps, 2.0/(3-e0))
                nn = sm.shape[0] + sm.shape[1]
                nlimit = int(math.ceil(nn**(1/ep)))
                cutrows = rows[:min(jr,nlimit)]
            else:
                cutrows = rows[:jr]

            CU.append(cutrows)

        self.CU = np.array(CU)
        self.CV = np.array(CV)
        return

    def initfastgreedy2D(self, numSing, rbd, eps=1.6):
        'rbd threshold that cut the singular vecotors, default is avg'
        'parameters for fastgreedy'
        u, s, vt = slin.svds(self.graphr.astype(np.float64), k=numSing, which='LM')
        #revert to make the largest singular values and vectors in the front
        u = np.fliplr(u)
        vt = np.flipud(vt)
        s = s[::-1]
        self.U = []
        self.V = []
        self.CU = []
        self.CV = []
        for i in range(self.numSing):
            ui = u[:, i]
            vi = vt[i, :]
            si = s[i]
            if abs(max(ui)) < abs(min(ui)):
                ui = -1*ui
            if abs(max(vi)) < abs(min(vi)):
                vi = -1*vi
            if type(rbd) is float:
                sqrtSi = math.sqrt(si)
                ui *= sqrtSi
                vi *= sqrtSi
                rbdrow, rbdcol = rbd, rbd
            elif rbd == 'avg':
                rbdrow = 1.0/math.sqrt(self.nU)
                rbdcol = 1.0/math.sqrt(self.nV)
            else:
                print('unkown rbd {}'.format(rbd))
            rows = np.argsort(-ui, axis=None, kind='quicksort')
            cols = np.argsort(-vi, axis=None, kind='quicksort')
            for jr in range(len(rows)):
                r = rows[jr]
                if ui[r] <= rbdrow:
                    break
            self.avgexponents.append(math.log(jr, self.nU))
            if self.nU > 5e5:
                e0=self.e0
                ep = max(eps, 2.0/(3-e0))
                nn = self.nU + self.nV
                nlimit = int(math.ceil(nn**(1.0/ep)))
                cutrows = rows[:min(jr,nlimit)]
            else:
                cutrows = rows[:jr]
            for jc in range(len(cols)):
                c = cols[jc]
                if vi[c] <= rbdcol:
                    break
            cutcols = cols[:jc]
            'begin debug'
            self.U.append(ui)
            self.V.append(vi)
            'end debug'
            self.CU.append(cutrows)
            self.CV.append(cutrows)

        self.CU = np.array(self.CU)
        self.CV = np.array(self.CV)
        return

    def qfunc(self, ratios, fbs=None, scale='exp', numratios=1):
        if self.aggmethod == 'rank':
            'do not use qfun if it is rank aggregation'
            return ratios

        if self.suspbd <= 0.0:
            greatbdidx = ratios > 0.0
        else:
            greatbdidx = ratios >= self.suspbd
            lessbdidx = ratios < self.suspbd
            'picewise q funciton if < suspbd, i.e. epsilon1'
            ratios[lessbdidx] = 0.0
        'picewise q funciton if >= suspbd, i.e. epsilon1'
        if scale == 'exp':
            ratios[greatbdidx] = self.expbase**(ratios[greatbdidx]-numratios)
        elif scale == 'pl':
            ratios[greatbdidx] = ratios[greatbdidx]**self.b
        elif scale == 'lin':
            ratios[greatbdidx] = np.fmax(self.b*(ratios[greatbdidx]-1)+1, 0)
        else:
            print('unrecognized scale: ' + scale)
            sys.exit(1)
        return ratios

    def setdeltapimsusp(self, z, ysuspusers, add):
        if Ptype.ts in self.ptype:
            self.tspim.deltasuspects(z, ysuspusers, add)
        if Ptype.rate in self.ptype:
            self.ratepim.deltasuspects(z, ysuspusers, add)
        return

    def removecurrentblock(self, rows):
        '''it is for find second block, remove rows from
           self.graph, self.matricizetenor
        '''
        print('removing {} rows from graph'.format(len(rows)))
        lilm = self.graph.tolil()
        lilm[rows,:]=0
        self.graph=lilm.tocoo()
        self.graphc= lilm.tocsc()
        self.graphr = self.graph.tocsr()

        if self.matricizetenor is not None:
            print('removing {} rows from tensor'.format(len(rows)))
            lilmm = self.matricizetenor.tolil()
            lilmm[rows,:] = 0
            self.matricizetenor = lilmm.tocoo()
        return

    #@profile
    def fastgreedy(self):
        'adding and deleting greed algorithm'
        'No Need: user order for r with obj fuct'
        self.fastlocalbest = []
        self.fastbestvx = 0
        self.fastbestA, self.fastbestfbs, self.fastbestbsusps = \
                np.zeros(self.nU), np.zeros(self.nV), np.zeros(self.nV)
        for k in range(self.numSing):
            print('process {}-th singular vector'.format(k+1))
            lenCU = len(self.CU[k])
            if lenCU == 0:
                continue
            print('*** *** shaving ...')
            A0 = np.zeros(self.nU, dtype=int)
            A0[self.CU[k]]=1 #shaving from sub singluar space
            self.start(A0, ptype=self.ptype)
            self.greedyshaving()
            print('*** *** shaving opt size: {}'.format(sum(self.bestA)))
            print('*** *** shaving opt value: {}'.format(self.bestvx))
            if self.fastbestvx < self.bestvx:
                self.fastbestvx = self.bestvx
                self.fastbestA = np.array(self.bestA)
                self.fastbestfbs = np.array(self.bestfbs)
                self.fastbestbsusps = np.array(self.bestbsusps)
                print('=== === improved opt size: {}'.format(sum(self.fastbestA)))
                print('=== === improved opt value: {}'.format(self.fastbestvx))

            brankscores = np.multiply(self.bestbsusps, self.bestfbs)
            A = self.bestA.nonzero()[0]
            self.fastlocalbest.append((self.bestvx, (A, brankscores)))
            'clear shaving best'
            self.bestvx = 0

        self.bestvx, self.bestA, self.bestfbs, self.bestbsusps = \
                    self.fastbestvx, self.fastbestA, \
                    self.fastbestfbs, self.fastbestbsusps
        return

    def drawObjectiveCurve(self, outfig):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(self.vxs, '-')
        plt.title('The convergence curve of simulated anealing.')
        plt.xlabel('# of iterations')
        plt.ylabel('objective value')
        if outfig is not None:
            fig.savefig(outfig)
        return fig


def holoscope_interface(wmat, alg, ptype, qfun, b, rateprop=None, tsprop=None,
              tunit='s', numSing=10, nblock=1, eps=1.6):
    '''
    The interface of HoloScope algorithm for external use
    Parameters
    ----------
    wmat: str or sparse matrix
        If it is str, wmat is the input file name. We load the file into sparse
        matrix. If it is sparse matrix, we just use wmat.
    alg: str
        which algorithm you are going to use. You can choose 'greedy' for
        synthetic data (#rows+#cols<10000); or 'fastgreedy' for any size of data
        sets.
    ptype: list
        contains which attributes the algorithm is going to use. The hololisc
        use of all siginals is [Ptype.freq, Ptype.ts, Ptype.rate]
    qfun: str
        which kind of qfun the algorithm uses, choosing from 'exp' for
        exponential (recommended), 'pl' for power-law, 'lin' for linear
    b: float
        The base of exponetial qfun, or the exponent of power-law qfun, or
        absolute slope of linear qfun
    rateprop: str or STensor or None
        The file name with path for user-object rating sequences. The file
        format is that each line looks like 'userid-objectid:#star1 #star2
        ...\n'.
        If it is STensor, then rateprop contains (userid, objectid, #star) --> freq
    tsprop: str or None
        The file name with path for user-object timestamp sequences. The file
        format is that each line looks like 'userid-objectid:t1 t2 ...\n'
        If it is STensor, then rateprop contains (userid, objectid, tsbin) --> freq
    tunit: str (only support 's' or 'd') or None
        The time unit of input time
        e.g. in amazon and yelp data, the time is date, i.e. tunit='d'.
             We use # of days (integer) from the earlest date as input
        It does not need if tsprop is STensor
    numSing: int
        The number of first left singular vectors used in our algorithm
    nblock: int
        The number of block we need from the algorithm
    eps: float
        It gives the approximate level cut for singular vectors, which
        is a trade-off parameter for efficency and accuracy. Usually eps
        is between (1.5, 2], and the complexity reduce from the quadratic in number of
        nodes to the near linear in number of edges.
    Return
    ---------
    (gbestvx, (gsrows, gbscores)), opt
        Block (gsrows, gbscores) has the best objective values 'gbestvx' among
	*nblock* blocks.
	gbestvx: float
            the best objective value of the *nblock* blocks.
        gsrows: list
            is the list of suspicious rows.
        gbscores: list
            is the suspicoius scores for every objects. The index is object id,
            and value is the score. With the scores, you can get the suspicious rank
        of the objects.
        opt: instance of HoloScopeOpt class
            the class instance which contains all the *nblock* blocks in opt.nbests.
            opt.nbests: list
                This is the list contains *nblock* solutions in the form of
                tuple, i.e., (opt.bestvx, (srows, bscores))
    '''
    print('initial...')
    if sci.sparse.issparse(wmat) is False and os.path.isfile(wmat):
        # sm = loadedge2sm(wmat, coo_matrix, weighted=True, idstartzero=True)
        # remove unexpected parameter 'weighted'
        sm = loadedge2sm(wmat, coo_matrix, idstartzero=True)
    else:
        sm = wmat.tocoo()
    inprop = 'Considering '
    if Ptype.freq in ptype:
        inprop += '+[topology] '
    if Ptype.ts in ptype:
        inprop += '+[timestamps] '
        #consider sdrop by default when Ptype.ts
        inprop += '+[sudden drop]'
    else:
        tsprop=None
    if Ptype.rate in ptype:
        inprop += '+[rating i.e. # of stars] '
    else:
        rateprop = None
    print(inprop)

    opt = HoloScopeOpt(sm, qfun=qfun, b=b, tsprop=tsprop, tunit=tunit,
            rateprop=rateprop)
    opt.nbests=[]
    opt.nlocalbests=[] #mainly used for fastgreedy
    gsrows,gbscores,gbestvx = 0,0,0
    for k in range(nblock):
        start_time = time.clock()
        if alg == 'greedy':
            n1, n2 = sm.shape
            if n1 + n2 > 1e4:
                print('[Warning] alg {} is slow for size {}x{}'\
                        .format(alg, n1, n2))
            A = np.ones(opt.nU,dtype=int)
            print('initial start')
            opt.start(A, ptype=ptype)
            print('greedy shaving algorithm ...')
            opt.greedyshaving()
        elif alg == 'fastgreedy':
            print("""alg: {}\n\t+ # of singlular vectors: {}\n""".format(alg, numSing))
            print('initial start')
            opt.initfastgreedy( ptype, numSing, eps=eps )
            print("::::Finish Init @ ", time.clock() - start_time)
            print('fast greedy algorithm ...')
            opt.fastgreedy()
            opt.nlocalbests.append(opt.fastlocalbest)
        else:
            print('No such algorithm: '+alg)
            sys.exit(1)

        print("::::Finish Algorithm @ ", time.clock() - start_time)

        srows = opt.bestA.nonzero()[0]
        bscores = np.multiply(opt.bestfbs, opt.bestbsusps)
        opt.nbests.append((opt.bestvx, (srows, bscores)))
        gsrows, gbscores, gbestvx = (srows,bscores,opt.bestvx) \
                if gbestvx < opt.bestvx  else (gsrows, gbscores, gbestvx)
        if k < nblock-1:
            opt.removecurrentblock(srows)

    #levelcols = score_level_objects( gbscores )
    nnzcols = nonzero_objects( gbscores )

    #print('global best size: nodes', len(gsrows), len(nnzcols), 'with camouflage.')

    #print('global best value ', gbestvx)

    return ((gsrows, nnzcols), gbestvx), gbscores[nnzcols], opt


class HoloScope( DMmodel ):
    '''Anomaly detection base on contrastively dense subgraphs, considering
    topological, temporal, and categorical (e.g. rating scores) signals, or
    any supported combinations.

    Parameters
    ----------
    graph: Graph
        Graph instance contains adjency matrix, and possible multiple signals.
    alg: string options ['fastgreedy' | 'greedy' ]
        The algorithm used for detect dense blocks. You can choose 'greedy' for
        synthetic data (#rows+#cols<10000); or 'fastgreedy' for any size of data
        sets.
        Default is 'fastgreedy' which
        uses main (with first several large singular values) truncated singular vectors
        to find dense blocks. alg is used with eps as truncation factor, and
        numSing as number of first large singular vectors.
    eps: float
        It gives the approximate level cut for singular vectors, which
        is a trade-off parameter for efficency and accuracy. Usually eps
        is between (1.5, 2], and the complexity reduce from the quadratic in number of
        nodes to the near linear in number of edges.
        Larger eps gives faster detection, but may miss the denser blocks.
        Default is 1.6.
    numSing: int
        The number of first large left singular vectors used in our algorithm
    qfun: string options ['exp' | 'pl' | 'lin']
        which kind of qfun the algorithm uses, choosing from 'exp' for
        exponential (recommended), 'pl' for power-law, 'lin' for linear
        Default is 'exp'.
    b: float
        The base of exponetial qfun, or the exponent of power-law qfun, or
        absolute slope of linear qfun
        Default is 32.
    '''
    def __init__(self, graph, **params):
        self.graph = graph
        self.alg = param_default(params, 'alg', 'fastgreedy')
        self.eps = param_default(params, 'eps', 1.6)
        self.numSing = param_default(params, 'numSing', 10)
        self.qfun = param_default(params, 'qfun', 'exp')
        self.b = param_default(params,'b', 32)

    def __str__(self):
        return str(vars(self))


    def run(self, k:int=1, level:int=0, eps:float = 1.6):
        '''run with how many blocks are output.
        Parameters:
        --------
        nblock: int
            The number of block we need from the algorithm
        level: int
            The level of signals used for anomaly detection. Choose in [0 | 1 | 2 |
            3]. 0: topology only. 1: topology with time. 2: topology with category
            (e.g. rating score). 3: all three.
            Default is 0.
        '''

        if eps != 1.6:
            epsuse=1.6
        else:
            epsuse = self.eps
        self.level = level
        nprop = self.graph.nprop
        graph = self.graph

        tsprop, rateprop = None, None
        if self.level == 0:
            ptype=[Ptype.freq]
        elif self.level ==1:
            ptype=[Ptype.freq, Ptype.ts]
            if nprop < 1:
                raise Exception("Error: at least 3-mode graph tensor is needed for level 1")
            tsprop = graph.get_time_tensor()
        elif self.level == 2:
            ptype = [Ptype.freq, Ptype.rate]
            if nprop < 1:
                raise Exception("Error: at least 3-mode graph tensor is needed for level 2")
            "The order of mode in graph tensor for categorical bins if exit, start from zero."
            modec = 3 if nprop > 1 else 2
            rateprop = graph.get_one_prop_tensor(modec)
        elif self.level == 3:
            ptype = [Ptype.freq, Ptype.ts, Ptype.rate]
            if nprop < 2:
                raise Exception("Error: at least 4-mode graph tensor is needed for level 3")
            tsprop = graph.get_time_tensor()
            modec=3
            rateprop = graph.get_one_prop_tensor(3)
        else:
            print("Warning: no run level ",self.level,", use level 0 instead!")
            ptype=[Ptype.freq]

        bdres = holoscope_interface(graph.sm.astype(float),
                self.alg, ptype, self.qfun, self.b,
                tsprop=tsprop, rateprop=rateprop,
                nblock=k, eps=epsuse, numSing=self.numSing)

        nres = []
        opt = bdres[-1]
        for nb in range(k):
            res = opt.nbests[nb]
            print('block{}: \n\tobjective value {}'.format(nb + 1, res[0]))
            levelcols = score_level_objects(res[1][1])
            nnzcols = nonzero_objects(res[1][1])
            rows = res[1][0]
            nleveledges = graph.get_subgraph_nedges( rows, levelcols )
            nedges = graph.get_subgraph_nedges( rows, nnzcols )
            print( '\tNode size: {} x {}, edge size {}'.format(len(rows),
                len(levelcols), nleveledges) )
            print( '\tRow and nonzero columns {} x {}, edge size: {} with camouflage.'.format(
                len(rows), len(nnzcols), nedges) )
            nres.append( ( (rows, levelcols), res[0], nnzcols, res[1][1][nnzcols] ) )

        self.nres = nres

        return nres

    def anomaly_detection(self, k:int=1, eps:float = 1.6):
        return self.run(k=k, eps=eps)

    def save(self, outpath):
        import pickle
        out = open(outpath,'wb')
        pickle.dump(self.nres, out)
        pass


