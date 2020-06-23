#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   anomaly_detection.py
@Desc    :   Implementation for anomaly detection task.
'''

# here put the import lib
from ._task import Task


class AnomalyDetection(Task):
    '''Implementation for anomaly detection task.
    '''

    def run(self):
        return self.policy.anomaly_detection(self.params)
