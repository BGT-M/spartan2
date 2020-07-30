#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   timeseries.py
@Desc    :   Definition of timeseries structure.
'''

# here put the import lib
from . import DTensor


class Timeseries:
    def __init__(self, time_tensor: DTensor, val_tensor: DTensor, labels: list = None, freq: int = 1, startts: int = 0):
        self.freq = freq
        self.val_tensor = val_tensor.T
        self.dimension, self.length = self.val_tensor.shape
        if labels is None:
            self.labels = ['dim_' + str(i) for i in range(self.dimension)]
        else:
            self.labels = labels
        if time_tensor is None:
            self.startts = startts
            import numpy as np
            self.time_tensor = DTensor.from_numpy(np.linspace(0, 1 / self.freq * self.length, self.length))
        else:
            self.startts = time_tensor[0]
            self.freq = (self.length - 1) / (time_tensor[-1] - time_tensor[0])
            self.time_tensor = time_tensor

    def __len__(self):
        self.length = self.time_tensor.__len__()[1]
        return self.length

    def __str__(self):
        ''' description of STTimeseries object
        '''
        _str = f'''
            Time Series Object
            Dimension Size: {self.dimension}
            Length: {self.length}
            Time Length: {format(self.length / self.freq, '.2f')}
            Frequency: {self.freq}
            Start Timestamp: {self.startts}
            Labels: {', '.join([str(x) for x in self.labels])}
        '''
        return _str

    def __copy__(self):
        import copy
        time_tensor = copy.copy(self.time_tensor)
        val_tensor = copy.copy(self.val_tensor).T
        labels = copy.copy(self.labels)
        return Timeseries(time_tensor, val_tensor, labels)

    def resample(self, resampled_freq: int, inplace: bool = False):
        _self = self._handle_inplace(inplace)
        # TODO
        pass

    def _handle_inplace(self, inplace):
        ''' return a new object or origin object
        '''
        if inplace:
            _self = self
        else:
            import copy
            _self = copy.copy(self)
        return _self
