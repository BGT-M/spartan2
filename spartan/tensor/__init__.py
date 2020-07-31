#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Desc    :   Import part and configuration part, including names exposed to spartan.
'''

# here put the import lib

from spartan.backend import STensor, DTensor

from .tensor import TensorData
from .tensor import TensorData, TensorStream
from .timeseries import Timeseries
from .graph import Graph

__all__ = [
    'TensorData', 'TensorStream', 'Timeseries', 'Graph'
]
