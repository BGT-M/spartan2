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
                if pair not in self.iatpaircount:
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
        from collections import Counter
        count_list = {}
        for pair in iatpairs:
            if pair in count_list:
                pair_counts = count_list[pair]
            else:
                if pair in self.iatpair_user:
                    pair_counts = Counter(self.iatpair_user[pair])
                    count_list[pair] = pair_counts
                else:
                    pair_counts = None
                    
            for usr in pair_counts:
                if usr in self.usrdict:
                    self.usrdict[usr] += pair_counts[usr]
                else:
                    self.usrdict[usr] = pair_counts[usr]
        
    def find_iqr_bound(self, x, k=1.5):
        x = np.array(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        bound = np.ceil(q75 + k * iqr)
        return bound
    
    def find_3sigma_bound(self, x):
        x = np.array(x)
        right = x.mean() + 3 * x.std()
        return right
    
    def find_suspicious_users(self, type_='iqr'):
        if type_ == 'iqr':
            print('use iqr bound')
            bound = self.find_iqr_bound(list(self.usrdict.values()))
        elif type_ == '3sigma':
            print('use 3sigma bound')
            bound = self.find_3sigma_bound(list(self.usrdict.values()))
        else:
            print(f'cannot find type_:{type_}, type_ should be in ["iqr", "3sigma"]')
            return 
        usrlist = []
        for usr, count in self.usrdict.items():
            if count > bound:
                usrlist.append(usr)
        return usrlist
            

    def find_topk_user(self, k=-1):
        '''find Top-K users that have pairs in iatpairs ordered by decreasing frequency
        Parameters:
        --------
        :param k: int
            default: -1 , means return all user
            else return Top-k user
        '''
        usrdict = sorted(self.usrdict.items(), key=lambda item: item[1], reverse=True)
        usrlist = [usrcountpair[0] for usrcountpair in usrdict[:k]] 
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
        





