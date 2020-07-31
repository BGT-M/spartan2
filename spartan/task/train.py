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

    def run(self, *args, **kwargs):
        '''Call train function of selected model.

        If not implemented, raise an exception by calling parent run.
        '''
        if "train" in dir(self.model):
            return self.model.train(*args, **kwargs)
        else:
            return super().run("train")

    @classmethod
    def list_models(cls):
        '''List models which can do training task.
        '''
        for name in TrainPolicy.__members__:
            print(name)


class TrainPolicy(Enum):
    '''Registration for path of models who can do train task.
    '''
    BeatGAN = MODEL_PATH + '.beatgan'
