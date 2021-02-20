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

from ._model import PipeLine


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


# please register a new module in lexical (alpha-beta) order
BeatLex = partial(__call__, MODEL_PATH + ".beatlex")
BeatGAN = partial(__call__, MODEL_PATH + ".beatgan")
EagleMine = partial(__call__, MODEL_PATH + ".eaglemine")
EigenPulse = partial(__call__, MODEL_PATH + ".eigenpulse")
Eigenspokes = partial(__call__, MODEL_PATH + ".eigenspokes")
HoloScope = partial(__call__, MODEL_PATH + ".holoscope")
FlowScope = partial(__call__, MODEL_PATH + ".flowscope")
RPeak = partial(__call__, MODEL_PATH + ".rpeak")
DPGS = partial(__call__, MODEL_PATH + ".DPGS")
kGrass = partial(__call__, MODEL_PATH + ".kGS")
IAT = partial(__call__, MODEL_PATH + ".iat")
Fraudar = partial(__call__, MODEL_PATH + ".fraudar")
CubeFlow = partial(__call__, MODEL_PATH + ".CubeFlow")


__all__ = [
    'PipeLine',
    'BeatLex',
    'BeatGAN',
    'EagleMine',
    'EigenPulse',
    'Eigenspokes',
    'HoloScope',
    'FlowScope',
    'RPeak',
    'DPGS',
    "kGrass",
    'IAT',
    'Fraudar',
    'CubeFlow'
]
