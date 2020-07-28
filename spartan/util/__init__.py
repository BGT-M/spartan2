#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Desc    :   Import part and configuration part, including names exposed to spartan.
'''

# here put the import lib

from spartan.tensor import TensorData

from .ioutil import loadTensor
from .basicutil import set_trace, TimeMapper, StringMapper,\
    IntMapper, ScoreMapper

MODEL_PATH = 'spartan.model'

__all__ = [
    'loadTensor',
    'set_trace',
    'TimeMapper',
    'StringMapper',
    'IntMapper',
    'ScoreMapper'
]
