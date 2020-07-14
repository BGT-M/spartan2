#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   timeseries.py
@Desc    :   Definition of timeseries structure.
'''

# here put the import lib


class Timeseries:
    def __init__(self, time_tensor, val_tensor):
        self.val_tensor = val_tensor.T
        self.time_tensor = self.handle_time(time_tensor)

    def handle_time(self, time_tensor):
        if time_tensor is None:
            # TODO create a tensor by DTensor
            pass
        else:
            return time_tensor
