import functools
import numbers

import torch
import torch.sparse as sparse


class DTensor:
    def __init__(self, data):
        self.data = data


class STensor:
    def __init__(self, data):
        self.data = data
