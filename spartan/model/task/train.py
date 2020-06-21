#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Desc    :   Implementation for training task.
'''

# here put the import lib
from ._task import Task


class Train(Task):
    '''Implementation for training task.
    '''

    def run(self):
        return self.policy.train(self.params)
