#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Desc    :   Implementation for training task.
'''

# here put the import lib

from . import MODEL_PATH

from ._task import Task
from enum import Enum


class Train(Task):
    '''Implementation for training task.
    '''

    def run(self):
        '''Call train function of selected model.
        '''
        if "train" in dir(self.model):
            return self.model.train(self.params)
        else:
            return super().run()

    @classmethod
    def list_models(cls):
        '''List models which can do training task.
        '''
        for name in TrainPolicy.__members__:
            print(name)


class TrainPolicy(Enum):
    '''Registration for path of models who can do train task.
    '''
    pass
