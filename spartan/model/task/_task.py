#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _task.py
@Desc    :   Interface of base class Task.
'''

# here put the import lib
from .. import STTensor
from .._model import Model

_policy_dict = {

}

def get_policy(policy: str) -> Model:
    '''Get model class from policy string.

    Attributes
    ----------
    policy : str
        name of model
    
    Returns
    ----------
    ret_val : Model

    '''
    ret_val = None
    if not _policy_dict.__contains__(policy):
        raise KeyError(f"{policy} is not supported.")
    else:
        ret_val = _policy_dict[policy]
    return ret_val


class Task():
    '''Basic class of task entity.

    Bridge for data, model and parameters.

    Attributes
    ----------
    tensor : STTensor
        data object
    policy : str
        model object
    model_name : str
        model name string
    params : dict
        parameters dictionary, used for construction

    '''

    def __init__(self):
        '''Initialization function.'''
        self.tensor = None
        self.policy = None
        self.model_name = None
        self.params = None

    @classmethod
    def create(cls, tensor: STTensor, policy: str, model_name: str, **params) -> object:
        '''Create function, called by class.

        Instantiation of policy from string and creation of Task from other parameters.

        Parameters
        ----------
        tensor : STTensor
            data object
        policy : str
            model object
        model_name : str
            model name string
        params : dict
            parameters dictionary, used for construction

        Returns
        ----------
        Task
            object for task
        '''
        model_cls = get_policy(policy)
        model = model_cls(model_name)
        obj = cls()
        obj.tensor = tensor
        obj.policy = model
        obj.model_name = model_name
        obj.params = params
        return obj

    def run(self):
        '''Interface of run function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise NotImplementedError("Run function not implemented.")
