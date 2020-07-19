#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basicutil.py
@Desc    :   Basic util functions.
'''

# here put the import lib
from datetime import datetime
import time
import numpy as np

def set_trace( isset = True ):
    if isset is True:
        import ipdb; ipdb.set_trace()
    else:
        pass


class _Mapper:
    def __init__(self):
        pass
    def map(self, attrs):
        pass
    def revert(self, indices):
        pass

class TimeMapper(_Mapper):
    def __init__(self, timeformat='%Y-%m-%d', timebin=24*3600, mints=None):
        self.timeformat = timeformat
        self.timebin = timebin
        self.mints = mints

    def map(self, attrs):
        tindices = []
        set_trace(False)
        for t in attrs:
            da = datetime.strptime(t, self.timeformat)
            ts = time.mktime(da.timetuple())
            tindices.append( int(ts) )

        tindices = np.array(tindices)
        if self.mints is None:
            self.mints = tindices.min()

        tindices = (tindices - self.mints) // self.timebin
        return tindices

    def revert(self, indices):

        timestr = []
        tss = indices + self.mints + self.timebin//2
        for ts in tss:
            date_time = datetime.fromtimestamp(ts)
            d = date_time.strftime(self.timeformat)
            timestr.append(d)

        return np.array(timestr)


class ScoreMapper(_Mapper):
    ''' map score (int or float) into index (int)

    Parameters
    ----------
    scorebin : If scorebin is int, evenly divide from min(score) to max(score)
    into scorebin segments. If scorebin is a list, map score with list values
    as bounds.
    [0, 2.0, 3.0]

    '''
    def __init__(self, scorebin):
        self.scorebin = scorebin

    def map(self, attrs):
        pass

    def revert(self, indices):
        pass


class StringMapper(_Mapper):
    '''mapping the names or complex string ids of users and objects into
    indices. Note that if the matrix is homogenous, i.e. user-to-user
    relations, then use the same StringMapper instance for both
    user-to-user colums. The two mappers will share the same strdict, so
    mapping their ids into the sampe space.
    '''
    def __init__(self, strdict: dict ={}):
        self.strdict = strdict
        self.strids = []
        if len(self.strdict) >= 0 :
            self._append_string_ids()
        pass

    def _append_string_ids(self):
        for k, v in self.strdict.items():
            self.strids.append(k)

    def map(self, attrs):
        indices = {}
        for s in attrs:
            if s not in self.strdict:
                self.strdict[s] = len(self.strdict)
                self.strids.append(s) # add new ids
            indices = self.strdict[s]

        return indices

    def revert(self, indices):
        attrs = []

        for i in indices:
            attrs.append(self.strids[i])

        return attrs


class IntMapper(_Mapper):
    '''Shift the ints starting from zero
    '''
    def __init__(self, minint=None):
        self.minint = minint
        pass

    def map(self, attrs):
        if self.minint is not None:
            self.minint = min(attrs)
        indices = attrs-self.minint
        return indices

    def revert(self, indices):
        attrs = indices + self.minint
        return attrs

