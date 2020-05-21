class IAT:
    aggiat = {}  # key:user; value:iat list
    iatcount = {}  # key:iat; value:count

    def __init__(self, aggiat={}, iatcount={}):
        self.aggiat = aggiat
        self.iatcount = iatcount

    def calaggiat(self, aggts):
        'aggts: key->user; value->timestamp list'
        for k, lst in aggts.items():
            if len(lst) >= 2:
                self.aggiat[k] = []
                pre_ts = lst[0]
                for ts in lst[1:]:
                    iat = ts - pre_ts
                    self.aggiat[k].append(iat)
                    pre_ts = ts

    def caliatcount(self):
        for k, lst in self.aggiat.items():
            for iat in lst:
                if iat not in self.iatcount:
                    self.iatcount[iat] = 0
                self.iatcount[iat] += 1

    def findUsers(self, iats):
        usrlist = []
        for k, lst in self.aggiat.items():
            for iat in lst:
                if iat in iats:
                    usrlist.append(k)
        return usrlist

