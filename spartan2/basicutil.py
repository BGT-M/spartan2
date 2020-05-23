import numpy as np
import spartan2.ioutil as ioutil


class IAT:
    aggiat = {}  # key:user; value:iat list
    aggiatpair = {}  # key:user; value: (iat1, iat2) list
    iatpaircount = {}  # key:(iat1, iat2); value:count
    iatcount = {}  # key:iat; value:count

    def __init__(self, aggiat={}, aggiatpair={}, iatpaircount={}, iatcount={}):
        self.aggiat = aggiat
        self.aggiatpair = aggiatpair
        self.iatpaircount = iatpaircount
        self.iatcount = iatcount

    def calaggiat(self, aggts):
        'aggts: key->user; value->timestamp list'
        for k, lst in aggts.items():
            if len(lst) < 2:
                continue
            lst.sort()
            iat = np.diff(lst)
            self.aggiat[k] = iat

    def save_aggiat(self, outfile):
        ioutil.saveDictListData(self.aggiat, outfile)

    def load_aggiat(self, infile):
        self.aggiat = ioutil.loadDictListData(infile, ktype=str, vtype=int)

    def calaggiatpair(self):
        for k, lst in self.aggiat.items():
            pairs = []
            for i in range(len(lst) - 1):
                pair = (lst[i], lst[i + 1])
                pairs.append(pair)
            self.aggiatpair[k] = pairs

    def getiatpairs(self):
        xs, ys = [], []
        for k, lst in self.aggiat.items():
            for i in range(len(lst) - 1):
                xs.append(lst[i])
                ys.append(lst[i + 1])
        return xs, ys

    def caliatcount(self):
        for k, lst in self.aggiat.items():
            for iat in lst:
                if iat not in self.iatcount:
                    self.iatcount[iat] = 0
                self.iatcount[iat] += 1

    def caliatpaircount(self):
        for k, lst in self.aggiat.items():
            for i in range(len(lst) - 1):
                pair = (lst[i], lst[i+1])
                if pair not in self.iatcount:
                    self.iatpaircount[pair] = 0
                self.iatpaircount[pair] += 1

    'find users that have pairs in iatpairs'
    def find_iatpair_user(self, iatpairs):
        usrlist, pairset = [], set(iatpairs)
        for k, lst in self.aggiatpair.items():
            if len(set(lst) & pairset) != 0:
                usrlist.append(k)
        return usrlist