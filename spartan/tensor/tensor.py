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

    def toSTensor(self, hasvalue: bool = True, mappers: dict = {}):
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

        ##assert(attr.dtypes[0] is int and  attr.dtypes[1] is int)
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
