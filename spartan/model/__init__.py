from spartan.util import MODEL_PATH
from enum import Enum
from functools import partial


def __call__(policy, *args, **kwargs):
    import importlib
    model_cls = importlib.import_module(policy).__call__()
    return model_cls(*args, **kwargs)


BeatLex = partial(__call__, MODEL_PATH + ".beatlex")

__all__ = [
    'BeatLex'
]
