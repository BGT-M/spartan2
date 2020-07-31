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

    def __init__(self, model_name: str = "my_model", *args, **kwargs):
        '''Initialization function.'''
        self.model_name = model_name

    @classmethod
    def __create__(cls, data: object, *args, **kwargs):
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

    def __init__(self, tensor: object, *args, **kwargs):
        '''Only support construction by classmethod.
        '''
        super(DMmodel, self).__init__(*args, **kwargs)

    @classmethod
    def __create__(cls, tensor: object, *args, **kwargs) -> object:
        '''Interface of creation by class, overrided by subclasses.

        Parameters
        ----------
        tensor: object
            data object
        params: dict
            parameter dictionary

        Returns
        ----------
        _obj
            object for specific model
        '''
        _obj = cls(tensor, *args, **kwargs)
        return _obj

    def run(self, *args, **kwargs):
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

    def __init__(self, tensor: object, *args, **kwargs):
        '''Only support construction by classmethod.
        '''
        super(MLmodel, self).__init__(*args, **kwargs)

    @classmethod
    def __create__(cls, tensor: object, *args, **kwargs) -> object:
        '''Interface of creation by class, overrided by subclasses.

        Parameters
        ----------
        tensor: object
            data object
        params: dict
            parameter dictionary

        Returns
        ----------
        _obj
            object for specific model
        '''
        _obj = cls(tensor, *args, **kwargs)
        return _obj

    def fit(self, *args, **kwargs):
        '''Interface of fit function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("Fit function not implemented.")

    def predict(self, *args, **kwargs):
        '''Interface of predict function, overrided by subclasses.

        Raises
        ----------
        NotImplementedError
            when called, return not implemented error
        '''
        raise Exception("Predict function not implemented.")
