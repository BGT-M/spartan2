import sys
import numpy as np
import scipy as sci
from scipy.sparse import coo_matrix, csc_matrix
# from gendenseblock import *
from .mytools.ioutil import myreadfile
import math
from spartan.backend import STensor, DTensor

class MultiEedgePropBiGraph:
    def __init__(self, wadjm):
        self.wadjm = wadjm.tocsr().astype(np.float64) #weighted adjacent matrix
        self.nU , self.nV = wadjm.shape
        self.indegrees = self.wadjm.sum(0).getA1()
        self.inbd=2 # the objects that has at least 2 edges are considered

    #@profile
    def load_from_edgeproperty(self, profnm, mtype=coo_matrix,
                               dtype=int):
        'load the graph edge property, time stamps, ratings, or text vector'
        self.idstartzero = True #record for output recovery
        offset = -1 if self.idstartzero is False else 0
        'sparse matrix has special meaning of 0, so property index start from 1'
        self.eprop = [np.array([])] #make the idx start from 1 in sparse matrix
        with myreadfile(profnm, 'rb') as fin:
            idx=1
            x,y,data=[],[],[]
            for line in fin:
                um, prop = line.strip().split(':')
                u, m = um.split('-')
                u = int(u)+offset
                m = int(m)+offset
                x.append(u)
                y.append(m)
                data.append(idx) #data store the index of edge properties
                prop = np.array(prop.strip().split()).astype(dtype)
                self.eprop.append(prop)
                idx += 1
            fin.close()
            self.edgeidxm = mtype((data, (x,y)), shape=(max(x)+1, max(y)+1))
            self.edgeidxmr = self.edgeidxm.tocsr()
            self.edgeidxmc = self.edgeidxm.tocsc()
            self.edgeidxml = self.edgeidxm.tolil()
            self.edgeidxmlt = self.edgeidxm.transpose().tolil()
            self.eprop = np.array(self.eprop)
        pass

    def trans_array_to_edgeproperty(self, arr:STensor, mtype=coo_matrix, dtype=int):
        ''' transform the ndarray arr to a pmi matrix.
        Do not support idstartzero. Use id as it is.
        '''
        'index of (v, v, attribute), and frequency of it '
        coords, freqs = arr.coords, arr.data
        # todo: it is better to have reduce func to have all prop combine
        uvdict = {}
        eprop=[[]] # make the first start from 1
        for i in range(len(freqs)):
            u,v,p = coords[:, i]
            if (u,v) not in uvdict:
                uvdict[(u,v)] = len(uvdict) + 1 #start from 1
                eprop.append([])
            eprop[ uvdict[(u,v)] ] += [p]*freqs[i]

        x,y = np.array(list(uvdict.keys())).T
        data = list(uvdict.values())
        self.edgeidxm = mtype((data, (x,y)), shape=(max(x)+1, max(y)+1) )
        self.edgeidxmr = self.edgeidxm.tocsr()
        self.edgeidxmc = self.edgeidxm.tocsc()
        self.edgeidxml = self.edgeidxm.tolil()
        self.edgeidxmlt = self.edgeidxm.transpose().tolil()
        self.eprop = np.array(eprop)
        pass

    #@profile
    def setup_rate4all_sinks(self):
        '''set up the rating property for all sinks'''
        propvals=set() #vacabulary size or score space
        for vs in self.eprop:
            for v in set(vs):
                propvals.add(v)
        self.propvals = np.array(sorted(list(propvals)))
        'assume score is [1,5], and  arounding real scores into 3 catagories, '
        if not (max(propvals)==5 and min(propvals)>=1):
            print('Warning: rating scores are not in [1,5]. They are [{}]'.\
                    format(', '.join(map(str, propvals))))

        '(1, 1.5, 2), (2.5, 3, 3.5), (4, 4.5, 5)'
        for i in range(len(self.eprop)):
            if min(propvals)<1:
                self.eprop[i] = np.digitize(self.eprop[i], bins=[0,2.5,4,5.01])-1
            else:
                self.eprop[i] = np.digitize(self.eprop[i], bins=[1,2.5,4,5.01])-1
        allmlt = self.edgeidxmlt #all susp msg matrix
        'effect sinks'
        cols = np.argwhere(self.indegrees>=self.inbd).flatten()
        self.inbdcolset = set(cols)
        apv = {}  #all property values
        ahists={} #all histograms of sinks
        amean, avar = np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        for i in cols:
            aidx = allmlt.data[i]
            apvi = np.concatenate(self.eprop[aidx]) #no np.sort
            apv[i]= apvi
            amean[i] = apvi.mean()
            avar[i] = apvi.var()
            ahists[i] = np.bincount(apvi, minlength=3)
        self.amean, self.avar = amean, avar
        self.ahists = ahists
        self.apv = apv
        return

    #@profile
    def setup_ts4all_sinks(self, twait, bins='auto'):
        'calculate the one-time values for every sink, like bursting, dying, drop'
        maxts = [np.max(t) for t in self.eprop[1:]]
        self.endt = max(maxts)
        self.twait = twait
        allmlt = self.edgeidxmlt #all susp msg matrix
        'effect sinks'
        cols = np.argwhere(self.indegrees>=self.inbd).flatten()
        self.inbdcolset = set(cols)
        apv = {}
        awakburstpt, burstvals, burstslops, ainbursts={},{},{},{}
        dyingpt = {}
        dropslops, dropfalls=np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        amean, avar = np.zeros(self.nV, dtype=np.float64), \
                np.zeros(self.nV, dtype=np.float64)
        for i in cols:
            aidx = allmlt.data[i]
            aumts = np.concatenate(self.eprop[aidx]) #no sort
            apv[i]= aumts
            amean[i] = aumts.mean()
            avar[i] = aumts.var()
            'awaking bursting points and values, debugpt for debug'
            abpts, bvs, slops, debugpt = awakburstpoints_recur(aumts, bins=bins)
            awakburstpt[i], burstvals[i], burstslops[i] =abpts, bvs, slops
            cnts=[]
            for abpt in abpts:
                '#of edges involve in bursting'
                left, right = abpt
                cnt = ((aumts>=left) & (aumts<=right)).sum()
                cnts.append(cnt)
            ainbursts[i]=np.array(cnts)
            dropfall, dropt, slop = \
                burstmaxdying_recur(aumts, endt=self.endt, twait=self.twait, bins=bins)
            dyingpt[i] = dropt
            dropslops[i], dropfalls[i] = slop, dropfall

        self.amean, self.avar = amean, avar
        self.apv = apv
        self.awakeburstpt, self.burstvals, self.burstslops, self.ainbursts = \
                awakburstpt, burstvals, burstslops, ainbursts
        self.dyingpt, self.dropslops, self.dropfalls = \
                dyingpt, dropslops, dropfalls
        return

    'this is only called once, always put into the init/start func'
    #@profile
    def setupsuspects(self, users):
        self.suspuser = np.array(users)
        self.deltacols, self.delcols = [], set()
        if len(self.suspuser) ==0:
            self.spv = {}
            return
        #suspmlt = self.edgeidxmr[self.suspuser].transpose().tolil()
        suspmlt = self.edgeidxml[self.suspuser].transpose()
        colwsum = self.wadjm[self.suspuser].sum(0).getA1()
        cols = np.where(colwsum>= self.inbd)[0]
        cols = set(cols) & self.inbdcolset
        spv = {}
        for col in cols:
            spids = suspmlt.data[col]
            #property indices of suspect sink
            #only consider those objects have more than inbd edges with suspusers
            sumts = np.concatenate(self.eprop[spids])
            spv[col]=sumts
        self.spv = spv
        return

    'must be effecient, shared among rating, ts, text'
    #@profile
    def deltasuspects(self, z, yusers, add=True):
        self.suspuser = yusers
        zmat = self.edgeidxmr[z]
        cols = zmat.nonzero()[1]
        deltacols, delcols = [], set()
        i = -1
        for col in cols:
            i += 1
            if col not in self.inbdcolset:
                continue
            spid = zmat.data[i]
            if add:
                self.spv[col]= np.concatenate((self.spv[col],self.eprop[spid]))\
                        if col in self.spv else self.eprop[spid]
                deltacols.append(col)
            else:
                if col not in self.spv:
                    continue #donot added in the initial
                #minus
                self.spv[col] = list(self.spv[col])
                for e in self.eprop[spid]:
                    self.spv[col].remove(e)
                if len(self.spv[col])==0:
                    self.spv.pop(col, None)
                    delcols.add(col)
                else:
                    deltacols.append(col)

        self.deltacols, self.delcols = deltacols, delcols
        return

    #@profile
    def suspratedivergence(self, neutral=False, delta=False):
        '''calculate the diverse of ratings betwee A and U\A
           scaling=False
        '''
        if delta and hasattr(self, 'ratediv'):
            cols, delcols = self.deltacols, self.delcols
            ratediv = self.ratediv
            if len(self.spv) < 1:
                self.ratediv[0:]=0.0
                return self.ratediv
        else:
            cols, delcols = self.spv.keys(), set()
            ratediv =np.zeros(self.nV, dtype=float)
            self.maxratediv = 0

        for col in cols:
            if col in delcols:
                assert(col not in self.spv)
                ratediv[col] = 0
                continue
            rs = self.spv[col]
            shis=np.bincount(rs, minlength=3)
            ahis = self.ahists[col]
            ohis=ahis-shis
            shis, ohis = shis+1, ohis+1 #a kind of multinomial prior
            if neutral is False:
                'remove netrual 2.5, 3, 3.5'
                shis[1], ohis[1] = 0, 0
            #cal KL-divergence
            from scipy import stats
            kl = stats.entropy(shis, ohis)
            lenrs = len(rs)
            lenars = len(self.apv[col])
            ssum, osum = float(lenrs)+1, float(lenars-lenrs)+1
            bal = (min(ssum/osum, osum/ssum))
            ratediv[col]=kl*bal
            self.maxratediv = kl if self.maxratediv < kl else self.maxratediv

        self.ratediv = ratediv
        return self.ratediv

    #@profile
    def suspburstinvolv(self, multiburstbd=0.5, weighted=True, delta=False):
        '''calc how many points allocated in awake and burst period, over total
           number of U who involv in the burst
        '''
        if delta and hasattr(self, 'incurstcnt') and hasattr(self, 'inburstratio'):
            cols, delcols = self.deltacols, self.delcols
            inburstcnt, inburstratio = self.inburstcnt, self.inburstratio
        else:
            inburstcnt, inburstratio = \
                    np.zeros(self.nV, dtype=int), np.zeros(self.nV, dtype=float)
            cols, delcols = self.spv.keys(), set()

        for col in cols:
            if col in delcols:
                assert(col not in self.spv)
                inburstcnt[col], inburstratio[col] = 0, 0.0
                continue
            st = self.spv[col]
            abpts, bvs, slops, ainburst = self.awakeburstpt[col], \
                    self.burstvals[col], self.burstslops[col], self.ainbursts[col]
            'get the satisfied multiburst points'
            burstids = bvs/float(bvs[0]) >=  multiburstbd
            abpts, slops, bvs, ainburst = abpts[burstids], slops[burstids],\
                    bvs[burstids], ainburst[burstids]
            scnt, wscnt, wallcnt =0, 0, 0
            for i in range(len(abpts)):
                (left, right), sp, bv, acnt = abpts[i],slops[i], bvs[i], ainburst[i]
                '#susp users in burst'
                cnt1 = ((st >= left) & (st <= right)).sum()
                scnt += cnt1
                '#all users in burst'
                assert(acnt>=cnt1)
                if weighted is not False:
                    wscnt +=  cnt1 * sp * bv
                    wallcnt += acnt * sp * bv
                else:
                    wscnt += cnt1
                    wallcnt += acnt
            inburstcnt[col]=scnt
            inburstratio[col] = wscnt/float(wallcnt)

        self.inburstcnt = inburstcnt
        self.inburstratio =inburstratio
        return self.inburstcnt, self.inburstratio

#@profile
def awakburstpoints_recur(ts, bins='auto'):
    'recursive version'
    hts = np.histogram(ts, bins=bins)
    ys = np.append([0], hts[0]) #add zero, so 0 is allocated to lowest left bound
    ys = ys.astype(np.float64)
    xs = hts[1]
    abptidxs = []
    startidx = 0
    'recursively get the idx for awake and burst pts'
    recurFindAwakePt(xs, ys, start=startidx, abptidxs=abptidxs)
    if len(abptidxs)==0:
        return [], [0], [0], None
    'extend left bound by -1, since we added zero to histogram'
    abptextidxs, bvsrt, slops = sort_extendLeftbd(abptidxs, xs, ys)
    'convert abptext idx to bd value in xs'
    abpts = np.array([(xs[l], xs[r]) for l, r in abptextidxs])
    return abpts, bvsrt, slops, [abptidxs, abptextidxs]

#@profile
def sort_extendLeftbd(abptidxs, xs, ys):
    'sort bds by burst val, and extend the left bound of sorted awakeburst pts'
    bv=[ ys[r]-ys[l] for l, r in abptidxs] #use abdiff as bv
    abptys = sorted(zip(abptidxs, bv), key=lambda x:x[1], reverse=True)
    abptsrt, bvsrt = zip(*abptys)
    abptsrt = np.array(abptsrt)
    bvsrt = np.array(bvsrt)
    'calculate slop of bursting before extending'
    slops, diffs = [], []
    for l, r in abptsrt:
        slop = (ys[r]-ys[l])/float(xs[r]-xs[l])
        slops.append(slop)
    slops = np.array(slops)
    #diffs = np.array(slops)
    'extend left, if overlep keep that of higher burst val'
    for i in range(len(abptsrt)):
        nl, nr = max(abptsrt[i][0]-1,0), abptsrt[i][1]
        for j in range(i):
            pl, pr = abptsrt[j][0], abptsrt[j][1]
            if nr >= pr and nl < pr:
                nl = pr
            if nl <= pl and nr > pl:
                print('[Warning] extended a impossible bound')
                nr = pl #impossible case, recurFindAwakePt guarantees that
        abptsrt[i][0], abptsrt[i][1]=nl,nr #extend
    return abptsrt, bvsrt, slops

#@profile
def recurFindAwakePt(xs, ys, start=0, abptidxs=[]):
    if len(ys)<=1 or len(xs)<=1:
        return
    maxidx = np.argmax(ys)
    x0,y0,xm,ym = xs[0], ys[0], xs[maxidx], ys[maxidx]
    sqco = math.sqrt((ym-y0)**2 + (xm-x0)**2) #sqrt of coefficient
    xvec, yvec = xs[:maxidx], ys[:maxidx]
    dts = ((ym-y0)*xvec - (xm-x0)*yvec + (xm*y0 - ym*x0))/sqco
    xaidx = np.argmax(dts)
    abptidxs.append((xaidx+start, maxidx+start))
    'left'
    recurFindAwakePt(xs[:xaidx], ys[:xaidx], start=start, abptidxs=abptidxs)
    'right'
    diffyincrese = np.argwhere(np.diff(ys[maxidx:]) >0)
    if len(diffyincrese) > 0:
        turningptidx = diffyincrese[0,0]+maxidx
        recurFindAwakePt(xs[turningptidx:], ys[turningptidx:],
                         start = turningptidx + start,
                         abptidxs=abptidxs)
    return

def burstmaxdying_recur(ts, endt, twait=12*3600, bins='auto'):
    'endt is used to judge if the dying is caused by observation window'
    #todo consider weights
    hts = np.histogram(ts, bins=bins)
    xs = hts[1]
    ys = hts[0].astype(np.float64)
    if  len(ys) < 2:
        return 0, xs[0], 0
    maxts = max(ts)
    if maxts < endt - twait:
        ys = np.concatenate((ys, [0.0]))
    else:
        hadd = (ys[-1]+ys[-2])/2.0
        ys = np.concatenate((ys, [hadd]))

    maxdying=[0.0, 0.0, 0.0]
    recurFindMaxFallDying(xs, ys, maxdying)
    return maxdying

def recurFindMaxFallDying(xs, ys, maxdying):
    lenys = len(ys)
    if lenys < 2:
        return
    burstidx = lenys - np.argmax(ys[::-1]) -1 #the last max occurrence
    if ys[burstidx]-min(ys) < maxdying[0]:
        return
    if burstidx == lenys-1: #bursting at the end
        dyingidx = lenys-1
        slop = (ys[burstidx] - 0)/float(xs[-1]-xs[-2])
        fall = ys[burstidx]
    else:
        xm, ym, xe, ye = xs[burstidx], ys[burstidx], xs[-1], ys[-1]
        sqco = math.sqrt((ym-ye)**2 + (xm-xe)**2) #sqrt of coefficient
        xvec, yvec = xs[burstidx+1:], ys[burstidx+1:]
        dts = -((ym-ye)*xvec - (xm-xe)*yvec + (xm*ye - ym*xe))/sqco
        dts = np.absolute(dts)
        dyingidx = len(dts)-np.argmax(dts[::-1])-1 + burstidx+1
        slop = (ym - ys[dyingidx])/float(xs[dyingidx] - xm)#dyingidx alwasy >0
        if dyingidx == lenys -1:
            fall = ym # assume continue to fall to 0, keeping current slop
        else:
            fall = ym-ys[dyingidx]
    if fall > maxdying[0]:
        maxdying[0:3] = [fall, xs[dyingidx], slop]

    if dyingidx < lenys-1:
        'move to the right'
        subburstidx = np.argmax(ys[dyingidx:]) + dyingidx
        recurFindMaxFallDying(xs[subburstidx:], ys[subburstidx:], maxdying)
    if burstidx > 1:
        'move to the left'
        subdyingidx = np.argmin(ys[:burstidx])
        recurFindMaxFallDying(xs[:subdyingidx], ys[:subdyingidx], maxdying)
    return


# assistant function. check tunit when is called
def pim2tensorformat(tsfile, ratefile, tensorfile, tunit='s', tbins='h'):
    'convert the pim files: tsfile, ratefile into tensor file, i.e. tuples'
    rbins = lambda x: 0 if x<2.5 else 1 if x<=3.5 else 2 #lambda x: x
    propdict = {}
    with myreadfile(tsfile, 'rb') as fts, myreadfile(ratefile, 'rb') as frt,\
            open(tensorfile, 'wb') as fte:
        for line in fts:
            k,v = line.strip().split(':')
            propdict[k]=[v]
        for line in frt:
            k,v=line.strip().split(':')
            propdict[k].append(v)
        for k, vs in propdict.items():
            u, b = k.strip().split('-')
            tss = vs[0].strip().split(' ')
            tss = list(map(int, tss))
            if tunit == 's':
                'time unit is second'
                if tbins == 'h':
                    'time bin size is hour'
                    tss = np.array(tss, dtype=int)//3600
                elif tbins == 'd':
                    'time bin size is day'
                    tss = np.array(tss, dtype=int)//(3600*24)
            'no matter what the tunit is'
            if type(tbins) is int:
                tss = np.array(tss, dtype=int)//tbins
            tss = list(map(str, tss))
            'process ts'
            rts = vs[1].strip().split(' ')
            rts = list(map(float, rts))
            digrs = []
            for r1 in rts:
                r = rbins(r1)
                digrs.append(r)
            digrs = list(map(int, digrs))
            digrs = list(map(str, digrs))
            for i in range(len(tss)):
                fte.write(','.join((u, b, tss[i], digrs[i], '1')))
                fte.write('\n')
        fts.close()
        frt.close()
        fte.close()
    return

# assistant function. check tunit when is called
def tspim2tensorformat(tsfile, tensorfile, tunit='s', tbins='h',
                       idstartzero=True):
    offset = 0 if idstartzero else -1
    propdict = {}
    with myreadfile(tsfile, 'rb') as fts, myreadfile(tensorfile, 'wb') as fte:
        for line in fts:
            k,v = line.strip().split(':')
            propdict[k]=[v]
        for k, vs in propdict.items():
            u, b = k.strip().split('-')
            if idstartzero is False:
                u = str(int(u)+offset)
                b = str(int(b)+offset)
            tss = vs[0].strip().split(' ')
            tss = list(map(int, tss))
            if tunit == 's':
                'time unit is second'
                if tbins == 'h':
                    'time bin size is hour'
                    tss = np.array(tss, dtype=int)//3600
                elif tbins == 'd':
                    'time bin size is day'
                    tss = np.array(tss, dtype=int)//(3600*24)
            if type(tbins) is int:
                tss = np.array(tss, dtype=int)//tbins
            tss = list(map(str, tss))
            for i in range(len(tss)):
                fte.write(','.join((u, b, tss[i], '1')))
                fte.write('\n')
        fts.close()
        fte.close()
    return

