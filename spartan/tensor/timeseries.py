#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   timeseries.py
@Desc    :   Definition of timeseries structure.
'''

# here put the import lib
from . import DTensor


class Timeseries:
    def __init__(self, val_tensor: DTensor, time_tensor: DTensor = None, labels: list = None, freq: int = 1, startts: int = 0):
        """A class designed for time series data.

        Parameters
        ----------
        val_tensor : DTensor
            value tensor

        time_tensor : DTensor
            time tensor, default is None

        labels : list
            list of column names, default is None

        freq : int
            frequency of this series, default is 1

        startts : int
            start timetick, default is 0

        Examples
        ----------
        Timeseries can be constructed in many styles. Among all parameters, only val_tensor is necessary.

        Normally, val_tensor, time_tensor, and labels are passed in. Length of labels and val_tensor will be determined to be equal.
        And meanwhile, freq, startts will be invalid and inferred from time tensor.

        >>> Timeseries(val_tensor, time_tensor, labels=['col_1', 'col_2'])

        If labels are missing, program will defaultly assign a list of labels, as ['dim_1', 'dim_2', ...]

        >>> Timeseries(val_tensor, time_tensor)

        If time tensor is missed, program will automatically create a time tensor with parameter freq and startts.

        >>> Timeseries(val_tensor, freq=2, startts=100)
        """
        self.freq = freq
        self.val_tensor = val_tensor.T
        self.dimension, self.length = self.val_tensor.shape
        if labels is None:
            self.labels = ['dim_' + str(i) for i in range(self.dimension)]
        else:
            self.labels = list(labels)
        if time_tensor is None:
            self.startts = startts
            import numpy as np
            self.time_tensor = self.__init_time(self.val_tensor.shape[1], self.freq, self.startts)
        else:
            self.startts = time_tensor[0]
            self.freq = (self.length) / (time_tensor.max() - time_tensor.min())
            self.time_tensor = time_tensor

    def __len__(self):
        """Return and update length of time tensor as length of time series object.

        Returns
        ----------
        self.length
            length of time series object
        """
        self.length = self.time_tensor.__len__()[1]
        return self.length

    def __str__(self):
        """Return discription of time series object.

        Returns
        ----------
        _str : str
            discription of time series object
        """
        import pandas as pd
        _str = f"""
            Time Series Object
            Dimension Size: {self.dimension}
            Length: {self.length}
            Time Length: {round(self.time_tensor.max() - self.time_tensor.min(), 3)}
            Frequency: {round(self.freq, 3)}
            Start Timestamp: {round(self.startts, 3)}
            Labels: {', '.join([str(x) for x in self.labels])}
        """
        columns = ['Time']
        columns.extend(self.labels)
        print(pd.DataFrame(DTensor([self.time_tensor]).concatenate(self.val_tensor, axis=0)._data.T,
                           columns=columns))
        return _str

    def __copy__(self):
        """Return copy of time series object.

        Returns
        ----------
        object
            copy of time series object
        """
        import copy
        time_tensor = copy.copy(self.time_tensor)
        val_tensor = copy.copy(self.val_tensor).T
        labels = copy.copy(self.labels)
        return Timeseries(val_tensor, time_tensor, labels)

    def resample(self, resampled_freq: int, inplace: bool = False, show: bool = False):
        """Resample series data with a new frequency, acomplished on the basis of scipy.signal.sample.

        Parameters
        ----------
        resampled_freq : int
            resampled frequency

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new resampled object    
        """
        _ori_tensor = DTensor.from_numpy(self.val_tensor._data.copy())
        _ori_freq = self.freq
        _self = self.__handle_inplace(inplace)
        new_len = int(_self.length / _self.freq * resampled_freq)
        _self.val_tensor.resample(new_len, inplace=True)
        _self.__update_time(_self.val_tensor, resampled_freq, _self.startts)
        _self.__update_info(_self.labels, _self.time_tensor, _self.val_tensor)
        if show:
            from spartan.util.drawutil import plot_resampled_series
            plot_resampled_series(self, self.length, _self.length, _ori_freq, resampled_freq, _ori_tensor._data, _self.val_tensor._data, _self.startts)
        if not inplace:
            return _self

    def add_columns(self, attrs: list or str, values: [int, float, DTensor, list] = None, inplace: bool = False, show: bool = False):
        """Add new equal-length columns to Time series object.

        Parameters
        ----------
        attrs : list or str
            list or string of column names

        values: [int, float, DTensor, list]
            if type of values is int or float, function will create a equal-length ndarray filled with values
            if type of values is DTensor or list, function will judge the length, then add columns
            default is None

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with columns added
        """
        _self = self.__handle_inplace(inplace)
        _names_type = type(attrs)
        _values_type = type(values)
        if _names_type == str:
            if _values_type in [int, float]:
                _self.__add_single_column(attrs, values, _type='number')
            elif _values_type == DTensor:
                assert len(values.shape) == 1
                _self.__add_single_column(attrs, values, _type='tensor')
            elif _values_type == list:
                assert len(values) == 1
                _self.__add_single_column(attrs, values[0], _type='number')
            else:
                raise TypeError(f"Inappropriate values type of {type(values)}")
        elif _names_type == list:
            if _values_type == DTensor:
                assert values.shape[0] == len(attrs)
                _self.__add_multi_columns(attrs, values, _type='tensor')
            elif _values_type == list:
                assert len(values) == len(attrs)
                _value_type = type(values[0])
                if _value_type in [int, float]:
                    _self.__add_multi_columns(attrs, values, _type='number')
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def __add_multi_columns(self, attrs: list, tensor: DTensor, _type: str):
        """Private function for adding multiple columns, adding operation is finished by concatenate.

        Parameters
        ----------
        attrs : list
            list of column names

        tensor : DTensor
            tensor to be added

        _type : str
            if number, function will create an equal-length ndarray for DTensor
            if tensor, function will concatenate directly
        """
        if _type == 'number':
            import numpy as np
            tensor = DTensor.from_numpy(np.tile(np.array([tensor]).T, (1, self.length)))
        elif _type == 'tensor':
            tensor = tensor
        self.val_tensor.concatenate(tensor, inplace=True)
        self.labels.extend(attrs)
        self.dimension += len(attrs)

    def __add_single_column(self, attr: str, value: DTensor, _type: str):
        """Private function for adding single column, adding operation is finished by concatenate.

        Parameters
        ----------
        columns_names : str
            string of column name

        tensor : DTensor
            tensor to be added

        _type : str
            if number, function will create an equal-length ndarray for DTensor
            if tensor, function will concatenate directly
        """
        if _type == 'number':
            import numpy as np
            _data = DTensor.from_numpy(np.array([[value] * self.length]))
        elif _type == 'tensor':
            _data = value
        self.val_tensor.concatenate(_data, inplace=True)
        self.labels.append(attr)
        self.dimension += 1

    def concat(self, series: list or "Timeseries", inplace: bool = False, show: bool = False):
        """Concatenate self with another timeseries object with the same dimension.

        Parameters
        ----------
        series : list or Timeseries
            list of Timeseries object or Timeseries object

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with columns concatenated
        """
        _self = self.__handle_inplace(inplace)
        _type = type(series)
        if _type == list:
            _series = []
            for x in series:
                if type(x) == Timeseries:
                    _series.append(x.__copy__())
                else:
                    raise Exception(f'list contains non-Timeseries object')
            _self.__concat_several(_series)
        elif _type == Timeseries:
            _self.__concat_one(series.__copy__())
        _self.__update_time(_self.val_tensor, _self.freq, _self.startts)
        _self.__update_info(_self.labels, _self.time_tensor, _self.val_tensor)
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def __concat_one(self, serie: "Timeseries"):
        """Private function for concating single object.

        Parameters
        ----------
        serie : Timeseries
            serie to be concatenated
        """
        if not self.dimension == serie.dimension:
            raise Exception(f'dimension sizes are not the same with self {self.dimension} and object {serie.dimension}')
        for i in range(len(self.labels)):
            if not self.labels[i] == serie.labels[i]:
                raise Exception(f'{i}th dimension is not corresponding with self {self.labels[i]} and object {serie.labels[i]}')
        self.val_tensor.concatenate(serie.val_tensor, axis=1, inplace=True)

    def __concat_several(self, concated_series: list):
        """Private function for concating several objects.

        Parameters
        ----------
        concated_series : list
            list of timeseries object to be concatenated
        """
        for serie in concated_series:
            self.__concat_one(serie)

    def combine(self, series: "Timeseries" or list, inplace: bool = False, show: bool = False):
        """Combine self with columns of other timeseries objects which have the same length.

        Parameters
        ----------
        series : list or Timeseries
            list of Timeseries object or Timeseries object

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with columns combined
        """
        _self = self.__handle_inplace(inplace)
        _type = type(series)
        if _type == list:
            _series = []
            for x in series:
                if type(x) == Timeseries:
                    _series.append(x.__copy__())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            _self.__combine_several(_series)
        elif _type == Timeseries:
            _self.__combine_one(series.__copy__())
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def __combine_one(self, serie: "Timeseries"):
        """Private function for combining single object.

        Parameters
        ----------
        serie : Timeseries
            serie to be combined
        """
        if not self.freq == serie.freq:
            raise Exception(f'Frequency not matched, with {self.freq} and {serie.freq}')
        for label in serie.labels:
            if label in self.labels:
                for i in range(1, 10000):
                    if not (label + '_' + str(i)) in self.labels:
                        self.labels.extend([label + '_' + str(i)])
                        break
            else:
                self.labels.extend([label])
        self.dimension += serie.dimension
        self.val_tensor.concatenate(serie.val_tensor, axis=0, inplace=True)

    def __combine_several(self, combined_series: list):
        """Private function for combining several objects.

        Parameters
        ----------
        combined_series : list
            list of timeseries object to be combined
        """
        for serie in combined_series:
            self.__combine_one(serie)

    def extract(self, attrs: list or str = None, inplace: bool = False, show: bool = False):
        """Extract specific columns from self.

        Parameters
        ----------
        attrs : list or str
            list or string of column names, default is None

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with columns extracted
        """
        _self = self.__handle_inplace(inplace)
        _labels, _tensor = _self.__handle_attrs(attrs)
        _self.__update_info(_labels, _self.time_tensor, _tensor)
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def cut(self, start: int = None, end: int = None, attrs: list = None, form: str = 'point', inplace: bool = False, show: bool = False):
        """Cut sub sequence from chosen attribute columns.

        Parameters
        ----------
        start : int
            start timetick or point, default is None, cut from the very front position

        end : int
            end timetick or point, default is None, cut to the very last position

        attrs : list or str
            list or string of column names, default is None, return the all columns

        form : str
            type of start and end
            if 'point', start and end stand for positions of points
            if 'time', start and end stand for timeticks of points
            default is 'point'

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with tensor cut
        """
        _self = self.__handle_inplace(inplace)
        _labels, _tensor = _self.__handle_attrs(attrs)
        if form == 'point':
            start = start
            end = end
        elif form == 'time':
            if not start is None:
                start = int((start-_self.startts) * _self.freq)
            if not end is None:
                end = int((end-_self.startts) * _self.freq)
        else:
            raise Exception('Value of parameter form is not defined!')
        if start is None:
            start = 0
        if end is None:
            end = _self.length
        if start < 0 or end > _self.length:
            raise Exception(f'start pos: {start} with 0 and end pos {end} with {_self.length}')
        _self.time_tensor.cut(start, end, inplace=True)
        _tensor.cut(start, end, inplace=True)
        _self.__update_info(_labels, _self.time_tensor, _tensor)
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def normalize(self, attrs: list or str = None, strategy: str = 'minmax', inplace: bool = False, show: bool = False):
        """Normalize data in value_tensor.

        Parameters
        ----------
        attrs : list or str
            list or string of column names, default is None

        strategy : str
            strategy for normalization
            if 'minmax', normalize to [-1, 1]
            default is 'minmax'

        inplace : bool
            update origin object or return a new object, default is False

        show : bool
            if True, draw plot

        Returns
        ----------
        None or Timeseries object
            self or a new object with tensor normalized
        """
        _self = self.__handle_inplace(inplace)
        _labels, _tensor = _self.__handle_attrs(attrs)
        if strategy == 'minmax':
            _tensor = _self.__normalize_minmax(_tensor)
        else:
            raise TypeError(f'strategy: {strategy} is not supported.')
        _self.__update_info(_labels, _self.time_tensor, _tensor)
        _self.__handle_plot(show)
        if not inplace:
            return _self

    def __normalize_minmax(self, _tensor: DTensor):
        """Private function for normalize value tensor by minmax function.

        Parameters
        ----------
        _tensor : DTensor
            value tensor to be normalized by minmax function

        Returns
        ----------
        _tensor : DTensor
            normalized tensor
        """
        import numpy as np
        _min = np.tile(_tensor.min(axis=1).reshape((self.dimension, 1)), self.length)
        _max = np.tile(_tensor.max(axis=1).reshape((self.dimension, 1)), self.length)
        _middle = (_min + _max) / 2
        _tensor = (_tensor - _middle) / (_max - _min) * 2
        return _tensor

    def __handle_plot(self, show: bool):
        """Private function for plotting.

        Parameters
        ----------
        show : bool
            if True, call plot function in drawutils
        """
        from spartan.util.drawutil import plot_timeseries
        if show:
            plot_timeseries(self)

    def __handle_attrs(self, attrs: str or list):
        """Private function for checking labels and tensor of column names in attrs.

        Parameters
        ----------
        attrs : list or str
            list or string of column names

        Raises
        ----------
        TypeError:
            Raise if attrs is not str or list

        Exception:
            Raise if attrs has column names which are not in self.labels

        Returns
        ----------
        _labels, _tensor : list, DTensor
            Selected labels and value tensor
        """
        if type(attrs) == str:
            attrs = [attrs]
        elif type(attrs) == list:
            attrs = attrs
        elif attrs is not None:
            raise TypeError(f'Type of attrs: {type(attrs)}')
        if attrs is None:
            _labels = self.labels
            _tensor = self.val_tensor
        else:
            _labels = []
            _tensor = []
            for attr in attrs:
                if not attr in self.labels:
                    raise Exception(f'Attr {attr} is not found')
                _labels.append(attr)
                index = self.labels.index(attr)
                _tensor.append(self.val_tensor._data[index])
            _tensor = DTensor(_tensor)
        return _labels, _tensor

    def __handle_inplace(self, inplace: bool = False):
        """Private function for checking if a new object is needed

        Parameters
        ----------
        inplace : bool
            update origin object or return a new object, default is False

        Returns
        ----------
        None or Timeseries object
            self or a new object
        """
        if inplace:
            _self = self
        else:
            import copy
            _self = copy.copy(self)
        return _self

    def __update_info(self, _labels: list, _time: DTensor, _tensor: DTensor):
        """Update infomation of self from newly updated tensors.

        Parameters
        ----------
        _labels : list
            list of column names

        _time : DTensor
            time tensor

        _tensor : DTensor
            value tensor
        """
        assert len(_labels) == len(_tensor)
        self.labels, self.time_tensor = _labels, _time
        self.val_tensor, self.dimension = _tensor, len(_tensor)
        self.startts = self.time_tensor[0]
        self.length = self.val_tensor.shape[1]

    def __update_time(self, val_tensor: DTensor, freq: int, startts: int):
        """Update infomation of self from newly updated tensors.

        Parameters
        ----------
        val_tensor : DTensor
            value tensor

        freq : int
            frequency of series

        startts : int
            start time tick
        """
        _len = val_tensor.shape[1]
        self.length = _len
        self.time_tensor = self.__init_time(_len, freq, startts)
        self.freq = freq
    
    
    def __init_time(self, len: int, freq: int, startts: int):
        """Construct time tensor.

        Parameters
        ----------
        len : int
            length of time tensor

        freq : int
            frequency of series

        startts : int
            start time tick
        
        Returns
        ----------
        time_tensor : DTensor
            time tensor
        """
        import numpy as np
        time_tensor = DTensor.from_numpy(np.linspace(startts, 1 / freq * len + startts - 1, len))
        return time_tensor
