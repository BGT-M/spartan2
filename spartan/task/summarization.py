#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summarization.py
@Desc    :   Implementation for summarization task.
'''

# here put the import lib

from . import MODEL_PATH

from ._task import Task
from enum import Enum


class Summarization(Task):
    '''Implementation for summarization task.
    '''

    def run(self):
        '''Call summarization function of selected model.
        '''
        return self.model.summarization(self.params)

    @classmethod
    def list_models(cls):
        '''List models which can do summarization task.
        '''
        for name in SumPolicy.__members__:
            print(name)


class SumPolicy(Enum):
    '''Registration for path of models who can do summarization task.
    '''
    Beatlex = MODEL_PATH + ".beatlex.Beatlex"
