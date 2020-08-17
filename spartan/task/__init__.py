#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Desc    :   Import part and configuration part, including names exposed to spartan.
'''

# here put the import lib

from spartan.util import MODEL_PATH

from .anomaly_detection import ADPolicy, AnomalyDetection
from .forecast import ForePolicy, Forecast
from .summarization import SumPolicy, Summarization
from .train import TrainPolicy, Train

__all__ = [
    'ADPolicy', 'AnomalyDetection',
    'ForePolicy', 'Forecast',
    'SumPolicy', 'Summarization',
    'TrainPolicy', 'Train'
]
