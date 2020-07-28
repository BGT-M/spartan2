#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   TensorData.py
@Desc    :   Structure of input file.
'''

# here put the import lib

from ..util.basicutil import set_trace
from . import STensor, DTensor
import pandas as pd


class TensorData:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def toDTensor(self, hastticks: bool = True):
        if hastticks:
            time_tensor = DTensor(self.data.iloc[:, 0])
            val_tensor = DTensor(self.data.iloc[:, 1:])
        else:
            time_tensor = None
            val_tensor = DTensor(self.data)
        return time_tensor, val_tensor

    def toSTensor( self, hasvalue: bool = True, mappers:dict = {}):
        if hasvalue:
            value = self.data.iloc[:, -1]
            attr = self.data.iloc[:, :-1]
        else:
            value = 1
            attr = self.data

        for i in attr.columns:
            if i in mappers:
                colind = mappers[i].map(self.data.iloc[:,i])
                attr.iloc[:,i] = colind

        ##assert(attr.dtypes[0] is int and  attr.dtypes[1] is int)

        return STensor((attr.to_numpy().T, value.to_numpy()))


class TensorStream():
    def __init__(self, f, idxtypes, sep: str = ' ', mappers: dict = {}):
        '''
        :param filename: input data file
        :param tcolid: the column index of time
        :param mode: r/rb
        '''
        self.f = f
        self.sep = sep
        self.idxtypes = idxtypes
        self.mappers = mappers
        self.next_window_start_pos = self.f.tell()
        self.mappers = mappers

    def _get_file_end_pos(self):
        cur_pos = self.f.tell()
        self.f.seek(0, 2)
        end_pos = self.f.tell()
        self.f.seek(cur_pos, 0)
        return end_pos

    def fetch_slide_window(self, window: int = 10, stride: int = 5, ts_colidx: int = 0):
        end_pos = self._get_file_end_pos()
        if self.f.tell() == end_pos:
            raise Exception('all data has been processed')
        else:
            self.f.seek(self.next_window_start_pos, 0)
        self.next_window_start_pos = None

        tensorlist = []
        lineid = 0
        while True:
            cur_pos = self.f.tell()
            line = self.f.readline()
            coords = line.strip().split(self.sep)
            tline = []
            try:
                for i, tp in self.idxtypes:
                    if i == ts_colidx:
                        ts = tp(coords[i])
                        'map time during reading data'
                        if i in self.mappers:
                            ts = self.mappers[i].map([ts])[0]
                    tline.append(tp(coords[i]))
                tensorlist.append(tline)
            except Exception:
                raise Exception(f"The {i}-th col does not match the given type {tp} in line:\n{line}")
            if lineid == 0:
                start_ts = ts
                lineid += 1
                continue
            else:
                if ts - start_ts >= stride:
                    if self.next_window_start_pos is None:
                        self.next_window_start_pos = cur_pos
                    if ts - start_ts >= window:
                        tensorlist.pop(-1)
                        self.f.seek(self.next_window_start_pos, 0)
                        break
                if self.f.tell() == end_pos:
                    break
        tensorlist = pd.DataFrame(tensorlist)

        'map other columns, e.g. user, item'
        for i in tensorlist.columns:
            if i in self.mappers:
                colind = self.mappers[i].map(tensorlist.iloc[:, i])
                tensorlist.iloc[:, i] = colind
        return tensorlist
