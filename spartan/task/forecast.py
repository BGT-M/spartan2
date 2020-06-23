#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   forecast.py
@Desc    :   Implementation for forecast task.
'''

# here put the import lib
from ._task import Task


class Forecast(Task):
    '''Implementation for forecast task.
    '''

    def run(self):
        return self.policy.forecast(self.params)
