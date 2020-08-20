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


class PipeLine():
    def __init__(self, data, module_list, pipe_name="my_pipeline", *args, **kwargs):
        self.module_list = module_list
        self.data = data

    def run(self):
        
        data = self.data
        for module in self.module_list:
            model, params = module
            if not isinstance(model, tuple):
                model = model(data, **params)
                if isinstance(model, DMmodel):
                    data = model.run()
                elif isinstance(model, MLmodel):
                    model.fit()
                    data = model.predict()
            else:
                task, model = model
                task = task.create(data, model, **params)
                data = task.run()
        return data
        
class Generalmodel(Model):
    '''Interface for general model.
    '''
    def __init__(self, *args, **kwargs):
        '''Only support construction by classmethod.
        '''
        super(Generalmodel, self).__init__(*args, **kwargs)

    @classmethod
    def __create__(cls, *args, **kwargs) -> object:
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
        _obj = cls(*args, **kwargs)
        return _obj
