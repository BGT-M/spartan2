#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _task.py
@Desc    :   Interface of base class Task.
'''

# here put the import lib

import importlib


class Task():
    '''Basic class of task entity.

    Bridge for data, model and parameters.

    Attributes
    ----------
    tensor : object
        data object
    policy : str
        model object
    model_name : str
        model name string
    params : dict
        parameters dictionary, used for construction

    '''

    def __init__(self, *args, **kwargs):
        '''Initialization function.'''
        self.tensor = None
        self.policy = None
        self.model_name = None
        self.model = None
        self.params = None

    @classmethod
    def create(cls, tensor: object, policy: str, model_name: str, *args, **kwargs) -> object:
        '''Create function, called by class.

        Instantiation of policy from string and creation of Task from other parameters.

        Parameters
        ----------
        tensor : object
            data object
        policy : str
            model path, defined by an enum class
        model_name : str
            model name string
        params : dict
            parameters dictionary, used for construction

        Returns
        ----------
        Task
            object for task
        '''
        try:
            model_cls = importlib.import_module(policy.value).__call__()
        except Exception as e:
            print(e)
            raise Exception(f"{policy} Not Supported!")
        model = model_cls.__create__(tensor, *args, **kwargs)
        obj = cls()
        obj.tensor = tensor
        obj.model = model
        obj.policy = policy
        obj.model_name = model_name
        obj.params = kwargs
        return obj

    def run(self, __func__: str, *args, **kwargs):
        '''Interface of run function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise NotImplementedError(f"{self.policy} do not implement {__func__} function.")
