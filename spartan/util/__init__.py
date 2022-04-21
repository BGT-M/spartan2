#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Desc    :   Import part and configuration part, including names exposed to spartan.
'''

# here put the import lib

from spartan.tensor import TensorData, TensorStream

from .ioutil import loadTensor, File, loadTensorStream, loadFile2Dict, loadHistogram
from .basicutil import set_trace, TimeMapper, StringMapper,\
        IntMapper, ScoreMapper, DenseIntMapper
from .drawutil import plot_graph, plot_timeseries, drawEigenPulse, plot, \
        histogram_viz, clusters_viz, drawHexbin, drawRectbin, plot_dist
from .rect_histogram import RectHistogram
from .MinTree import MinTree

MODEL_PATH = 'spartan.model'

__all__ = [
    'loadTensor',
    'File',
    'loadTensorStream',
    'loadFile2Dict',
    'loadHistogram',
    'set_trace',
    'TimeMapper',
    'StringMapper',
    'IntMapper',
    'ScoreMapper',
    'DenseIntMapper',
    'plot',
    'plot_graph',
    'plot_timeseries',
    'plot_dist',
    'drawEigenPulse',
    'histogram_viz',
    'clusters_viz',
    'drawHexbin',
    'drawRectbin',
    'RectHistogram',
    'MinTree',
]
