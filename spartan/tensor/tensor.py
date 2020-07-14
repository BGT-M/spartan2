#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   TensorData.py
@Desc    :   Structure of input file.
'''

# here put the import lib

from . import STensor, DTensor
import pandas as pd


class TensorData:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def toDTensor(self, hastticks: bool = True):
        if hastticks:
            time_tensor = DTensor(self.data.iloc[:, 0])
            attr_tensor = DTensor(self.data.iloc[:, 1:])
        else:
            time_tensor = None
            attr_tensor = DTensor(self.data)
        return time_tensor, attr_tensor

    def toSTensor(self, hasvalue: bool = True):
        if hasvalue:
            value_tensor = STensor(self.data.iloc[:, -1])
            attr_tensor = STensor(self.data.iloc[:, :-1])
        else:
            value_tensor = None
            attr_tensor = STensor(self.data)
        return value_tensor, attr_tensor
