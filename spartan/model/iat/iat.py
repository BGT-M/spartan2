import numpy as np
from .._model import Generalmodel
# import spartan2.ioutil as ioutil
from spartan.util.ioutil import saveDictListData, loadDictListData


class IAT(Generalmodel):
    aggiat = {}  # key:user; value:iat list
    user_iatpair = {}  # key:user; value: (iat1, iat2) list
    iatpair_user = {}  # key:(iat1, iat2) list; value: user
    iatpaircount = {}  # key:(iat1, iat2); value:count
    iatcount = {}  # key:iat; value:count
    iatprob = {} # key:iat; value:probability
    usrdict = {} # key:usr, value:frequency

    def __init__(self, aggiat={}, user_iatpair={}, iatpair_user={}, iatpaircount={}, iatcount={}):
        self.aggiat = aggiat
        self.user_iatpair = user_iatpair
        self.iatpair_user = iatpair_user
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
        saveDictListData(self.aggiat, outfile)

    def load_aggiat(self, infile):
        self.aggiat = loadDictListData(infile, ktype=int, vtype=int)
    
    def get_iatpair_user_dict(self):
        'construct dict for iat pair to keys'
        for k, lst in self.aggiat.items():
            for i in range(len(lst) - 1):
                pair = (lst[i], lst[i + 1])
                if pair not in self.iatpair_user:
                    self.iatpair_user[pair] = []
                self.iatpair_user[pair].append(k)

    def get_user_iatpair_dict(self):
        for k, lst in self.aggiat.items():
            pairs = []
            for i in range(len(lst) - 1):
                pair = (lst[i], lst[i + 1])
                pairs.append(pair)
            self.user_iatpair[k] = pairs

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

        allcount = sum(self.iatcount.values()) # total sum of iat
        self.iatprob = {iat: self.iatcount[iat]/allcount for iat in self.iatcount.keys()} #cal probability of iat

    def caliatpaircount(self):
        for k, lst in self.aggiat.items():
            for i in range(len(lst) - 1):
                pair = (lst[i], lst[i+1])
                if pair not in self.iatcount:
                    self.iatpaircount[pair] = 0
                self.iatpaircount[pair] += 1

    def find_iatpair_user(self, iatpairs):
        'find users that have pairs in iatpairs'
        usrset = set()
        for pair in iatpairs:
            if pair in self.iatpair_user:
                usrlist = self.iatpair_user[pair]
                usrset.update(usrlist)
        return list(usrset)
    
    def get_user_dict(self, iatpairs):
        '''get users dict that have pairs in iatpairs ordered by decreasing frequency
        Parameters:
        --------
        :param iatpairs: dict
            iat pair returned by find_peak_rect function in RectHistogram class
        '''
        for pair in iatpairs:
            if pair in self.iatpair_user:
                usrlist = self.iatpair_user[pair]
                for usr in usrlist:
                    if usr in self.usrdict:
                        self.usrdict[usr] += 1
                    else:
                        self.usrdict[usr] = 1
        self.usrdict = sorted(self.usrdict.items(), key=lambda item:item[1], reverse=True)
    
    def find_topk_user(self, k=-1):
        '''find Top-K users that have pairs in iatpairs ordered by decreasing frequency
        Parameters:
        --------
        :param k: int
            default: -1 , means return all user
            else return Top-k user
        '''
        usrlist = [usrcountpair[0] for usrcountpair in self.usrdict[:k]] 
        return usrlist
        
    def drawIatPdf(self, usrlist: list, outfig=None):
        '''Plot Iat-Pdf line
        Parameters:
        --------
        :param usrlist: list
            Top-k user returned by find_iatpair_user_ordered function
        :param outfig: str
            fig save path
        '''
        iatset = set()
        for usrid in usrlist:
            iatset.update(self.aggiat[usrid])
        iatlist = list(iatset)
        iatlist.sort()

        import matplotlib.pyplot as plt
        fig = plt.figure()
        xs = iatlist
        ys = [self.iatprob[iat] for iat in iatlist]
        plt.plot(xs, ys, 'b')
        plt.xscale('log')
        plt.xlabel('IAT(seconds)')
        plt.ylabel('pdf')
        if outfig is not None:
            fig.savefig(outfig)
        return fig
        





