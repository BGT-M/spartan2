from spartan.util.basicutil import param_default

from .BeatGAN_CNN import BeatGAN_CNN
from .BeatGAN_RNN import BeatGAN_RNN


def __call__():
    return BeatGAN


class BeatGAN():
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError("Please select a network type.")

    @classmethod
    def __create__(cls, *args, **kwargs):
        if kwargs['network'] == 'CNN':
            return BeatGAN_CNN(*args, **kwargs)
        elif kwargs['network'] == 'RNN':
            return BeatGAN_RNN(*args, **kwargs)
        else:
            return cls(*args, **kwargs)
