#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _task.py
@Desc    :   Interface of base class Task.
'''

# here put the import lib
from .. import STTensor


class Task():
    '''Basic class of task entity.

    Bridge for data, model and parameters.

    Attributes
    ----------
    tensor : STTensor
        data object
    policy : str
        model object
    params : dict
        parameters dictionary, used for construction

    '''

    def __init__(self, tensor: STTensor, policy: str, **params):
        '''Initialization function.'''
        self.tensor = tensor
        self.policy = policy
        self.params = params

    def run(self):
        '''Interface of run function, override by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise NotImplementedError("Run function not implemented.")
