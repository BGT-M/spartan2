#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   forecast.py
@Desc    :   Implementation for forecast task.
'''

# here put the import lib

from . import MODEL_PATH

from ._task import Task
from enum import Enum


class Forecast(Task):
    '''Implementation for forecast task.
    '''

    def run(self, *args, **kwargs):
        '''Call forecast function of selected model.

        If not implemented, raise an exception by calling parent run.
        '''
        if "forecast" in dir(self.model):
            return self.model.forecast(*args, **kwargs)
        else:
            return super().run("forecast")

    @classmethod
    def list_models(cls):
        '''List models which can do forecast task.
        '''
        for name in ForePolicy.__members__:
            print(name)


class ForePolicy(Enum):
    '''Registration for path of models who can do forecast task.
    '''
    pass
