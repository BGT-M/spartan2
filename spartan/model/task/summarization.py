#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summarization.py
@Desc    :   Implementation for summarization task.
'''

# here put the import lib
from ._task import Task


class Summarization(Task):
    '''Implementation for summarization task.
    '''

    def run(self):
        return self.policy.summarization(self.params)
