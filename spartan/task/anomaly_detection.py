#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   anomaly_detection.py
@Desc    :   Implementation for anomaly detection task.
'''

# here put the import lib

from . import MODEL_PATH

from ._task import Task
from enum import Enum


class AnomalyDetection(Task):
    '''Implementation for anomaly detection task.
    '''

    def run(self, *args, **kwargs):
        '''Call anomaly detection function of selected model.

        If not implemented, raise an exception by calling parent run.
        '''
        if "anomaly_detection" in dir(self.model):
            return self.model.anomaly_detection(*args, **kwargs)
        else:
            return super().run("anomaly_detection")

    @classmethod
    def list_models(cls):
        '''List models which can do anomaly detection task.
        '''
        for name in ADPolicy.__members__:
            print(name)


class ADPolicy(Enum):
    '''Registration for path of models who can do anomaly detection task.
    '''
    # EagleMine = MODEL_PATH + ".eaglemine"
    EigenPulse = MODEL_PATH + ".eigenpulse"
    HoloScope = MODEL_PATH + ".holoscope"
    FlowScope = MODEL_PATH + ".flowscope"
    pass
