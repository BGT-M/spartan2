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
import numpy as np
from joblib import Parallel, delayed


class TensorData:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.labels = data.columns

    def toDTensor(self, hastticks: bool = True):
        if hastticks:
            time_tensor = DTensor(self.data.iloc[:, 0])
            val_tensor = DTensor(self.data.iloc[:, 1:])
        else:
            time_tensor = None
            val_tensor = DTensor(self.data)
        return time_tensor, val_tensor

    def toSTensor(self, hasvalue: bool = True, mappers: dict = {}):
        if hasvalue:
            value = self.data.iloc[:, -1]
            attr = self.data.iloc[:, :-1]
        else:
            value = pd.Series([1] * len(self.data))
            attr = self.data

        for i in attr.columns:
            if i in mappers:
                if isinstance(i, str):
                    colind = mappers[i].map(self.data.loc[:,i])
                    attr.loc[:,i] = colind
                else:
                    colind = mappers[i].map(self.data.iloc[:, i])
                    attr.iloc[:, i] = colind
        attr = attr.astype('int')
        return STensor((attr.to_numpy().T, value.to_numpy()))
        

    def do_map(self, hasvalue=True, mappers={}):
        if hasvalue:
            value = self.data.iloc[:, -1]
            attr = self.data.iloc[:, :-1]
        else:
            value = pd.Series([1] * len(self.data))
            attr = self.data

        for i in attr.columns:
            if i in mappers:
                colind = mappers[i].map(self.data.iloc[:, i])
                attr.iloc[:, i] = colind
        return attr.to_numpy(), value.to_numpy()

    def log_to_time(self, time_col: int or list = 0, group_col: int or list = 1, val_col: int or list = None, format: str = '%Y-%m-%d %H:%M:%S', bins: int = 10, range: tuple = None, inplace: bool = False):
        """ Transfer log data to time series data.

        Parameters:
        ----------
        time_col : int or list, optional
            position of time column, default is 0

        group_col : int or list, optional
            positions of columns used to group data, default is 1

        val_col : int or list, optional
            positions of columns to be aggregated, default is None, will aggregate all but time col and group col

        format : str, optional
            time format, default is '%Y-%m-%d %H:%M:%S'

        bins : int, optional
            number of equal width bins in the given range, default is 10
        """
        import time
        _data = self.data
        if type(group_col) != list:
            group_col = [group_col]
        if val_col is not None:
            if type(val_col) != list:
                val_col = [val_col]
        else:
            val_col = [x for x in _data.columns if x not in group_col + [time_col]]
        _data = _data.iloc[:, [time_col] + group_col + val_col]
        _data.iloc[:, time_col] = _data.iloc[:, time_col].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        if range is not None and len(range) == 2:
            _start, _end = range
            _data = _data.iloc[_start:_end, :]
        _min = _data.iloc[:, time_col].min()
        _max = _data.iloc[:, time_col].max()
        _interval = (_max - _min) / (bins)
        _data.iloc[:, time_col] = _data.iloc[:, time_col].apply(lambda x: int((x - _min) / _interval) * _interval + _min if int((x - _min) / _interval) < bins else int((x - _min) / _interval - 1) * _interval + _min)
        name_col = [time_col] + group_col
        grouped_data = _data.groupby(name_col).sum()
        import numpy as np
        time_df = pd.DataFrame({
            time_col: np.linspace(_min, _max, bins+1)
        })
        grouped_data = grouped_data.unstack(level=group_col)
        _ans = time_df.join(grouped_data, on=time_col, how='outer')
        if inplace:
            self.data = _ans
            self.labels = list(_ans.columns)
            self.labels.remove(self.labels[time_col])
        else:
            return _ans
    
    def to_aggts(self, data, time_col: int = 0, group_col: int or list = 1, inplace: bool = False):
        aggts = {} # final dict list for aggregating time series.
        for row in data:
            if len(group_col) == 1:
                key = row[group_col[0]]
            else:
                key = ','.join(np.array(row)[group_col])
            if key not in aggts:
                aggts[key] = []
            aggts[key].append(row[time_col])
        return aggts


class TensorStream():
    def __init__(self, f, col_idx: list = None, col_types: list = None, sep: str = ' ',
                 mappers: dict = {}, hasvalue: bool = True):
        self.f = f

        if col_types is None:
            if col_idx is None:
                self.idxtypes = None
            else:
                self.idxtypes = [(x, str) for x in col_idx]
        else:
            if col_idx is None:
                col_idx = [i for i in range(len(col_types))]
            if len(col_idx) == len(col_types):
                self.idxtypes = [(x, col_types[i]) for i, x in enumerate(col_idx)]
            else:
                raise Exception(f"Error: input same size of col_types and col_idx")
        self.sep = sep
        self.mappers = mappers
        self.mappers = mappers
        self.hasvalue = hasvalue

        self.lastwindow = []  # data of last window (lines)
        self.laststrides = []  # indices of strides of last window

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

        tensorlist = []
        lineid = 0

        while True:
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
            except Exception:
                raise Exception(f"The {i}-th col does not match the given type {tp} in line:\n{line}")
            if lineid == 0:
                start_ts = ts
            else:
                if ts - start_ts >= stride:
                    if len(self.lastwindow) == 0:  # first window
                        self.laststrides.append(lineid)
                        if ts - start_ts >= window:
                            self.lastwindow = tensorlist
                            break
                    else:
                        win_start_lineid = self.laststrides.pop(0)
                        for i in range(len(self.laststrides)):
                            self.laststrides[i] -= win_start_lineid
                        curwindow = []
                        curwindow.extend(self.lastwindow[win_start_lineid:])
                        curwindow.extend(tensorlist)
                        self.lastwindow = curwindow
                        self.laststrides.append(len(curwindow))
                        break
                if self.f.tell() == end_pos:
                    break
            tensorlist.append(tline)
            lineid += 1
        tensorlist = pd.DataFrame(tensorlist)

        'map other columns, e.g. user, item'
        for i in tensorlist.columns:
            if i in self.mappers:
                colind = self.mappers[i].map(tensorlist.iloc[:, i])
                tensorlist.iloc[:, i] = colind
        return tensorlist
