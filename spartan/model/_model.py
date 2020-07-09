#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _model.py
@Desc    :   Interface of class Model, DMmodel, MLmodel.
'''

# here put the import lib


class Model():
    '''Basic class of model entity.

    Attributes
    ----------
    model_name : str
        name of model
    '''

    def __init__(self, model_name: str = "my_model"):
        '''Initialization function.'''
        self.model_name = model_name

    @classmethod
    def __create__(cls, data: object, params: dict):
        '''Interface of creation by class, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("__create__ function not implemented.")


class DMmodel(Model):
    '''Interface for data mining model.
    '''

    def __init__(self, tensor, params):
        pass

    @classmethod
    def __create__(cls, tensor, params):
        _obj = cls(tensor, params)
        return _obj

    def run(self):
        '''Interface of run function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("Run function not implemented.")


class MLmodel(Model):
    '''Interface for machine learning model.
    '''

    def __init__(self, tensor, params):
        pass

    @classmethod
    def __create__(cls, tensor, params):
        _obj = cls(tensor, params)
        return _obj

    def fit(self):
        '''Interface of fit function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("Fit function not implemented.")

    def predict(self):
        '''Interface of predict function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("Predict function not implemented.")
