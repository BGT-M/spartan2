#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Desc    :   Import part and configuration part, including names exposed to spartan.
'''

# here put the import lib

from spartan.util import MODEL_PATH
from enum import Enum
from functools import partial


def __call__(policy: str, *args, **kwargs) -> object:
    '''Design for dynamic import of call by model.

    Use partial function to point out specific path of each model.

    Parameters
    ----------
    policy : str
        path of model

    Returns
    ----------
    model_obj
        object for model
    '''
    import importlib
    model_cls = importlib.import_module(policy).__call__()
    return model_cls.__create__(*args, **kwargs)


BeatLex = partial(__call__, MODEL_PATH + ".beatlex")
HoloScope = partial(__call__, MODEL_PATH + ".holoscope")
Summarize = partial(__call__, MODEL_PATH + ".summarize")
BeatGAN = partial(__call__, MODEL_PATH + ".beatgan")

__all__ = [
    'BeatLex',
    'HoloScope',
    'BeatGAN',
    'Summarize',
]
