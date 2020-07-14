#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   TensorData.py
@Desc    :   None
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


class Timeseries:
    def __init__(self, time_tensor, attr_tensor):
        self.attr_tensor = attr_tensor.T
        self.time_tensor = self.handle_time(time_tensor)
    
    def handle_time(self, time_tensor):
        if time_tensor is None:
            # TODO create a tensor by DTensor
            pass
        else:
            return time_tensor

class Graph:
    def __init__(self, value_tensor, attr_tensor):
        self.attr_tensor = attr_tensor
        self.value_tensor = self.handle_value(value_tensor)

    def handle_value(self, value_tensor):
        if value_tensor is None:
            return None
        else:
            return value_tensor
