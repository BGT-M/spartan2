# -*- coding:utf-8 -*-
# Authors: Quan Ding

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200


class STTimeseries:
    def __init__(self, time: int, attrlists: np.ndarray, attrlabels: list, freq: int = None, startts: int = None):
        ''' init STTimeseries object

        Args:
            time: time dimension of data
            attrlists: signal dimensions of data
            attrlabels: labels of data, positions corresponding to data
            freq: frequency of the signal, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, freq will not work and will be calculated by the time sequence
            startts: start timestamp, default is None
                if time is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, startts will not work and will be calculated by the time sequence
        '''
        self.length = len(attrlists[0])
        self.dimen_size = len(attrlists)
        self.time_origin = True if not len(time) == 0 else False
        if len(time) == 0:
            if freq is None:
                raise Exception('Parameter freq not provided')
            if startts is None:
                raise Exception('Parameter startts not provided')
            self.freq = freq
            self.startts = startts
            self.timelist = np.arange(startts, 1/freq*self.length, 1 / freq)
        else:
            self.timelist = time
            self.freq = int(len(time) / (time[-1] - time[0]))
            self.startts = time[0]
        self.attrlists = attrlists
        self.attrlabels = attrlabels

    def __str__(self):
        ''' description of STTimeseries object
        '''
        _str = f'''
            Time Series Object
            Dimension Size: {self.dimen_size}
            Length: {self.length}
            Time Length: {format(self.length / self.freq, '.2f')}
            Origin Time Dimension: {self.time_origin}
            Frequency: {self.freq}
            Start Timestamp: {self.startts}
            Labels: {', '.join([str(x) for x in self.attrlabels])}
        '''
        return _str

    def __len__(self):
        ''' return length of time series
        '''
        return self.length

    def copy(self):
        ''' copy a new STTimeseries object from self

        Returns:
            STTimeseries object
        '''
        time = self.timelist.copy()
        attrlists = self.attrlists.copy()
        attrlabels = self.attrlabels.copy()
        return STTimeseries(time, attrlists, attrlabels)

    def show(self, chosen_labels: list = None):
        ''' draw series data with matplotlib.pyplot

        Args:
            chosen_labels: if not provided, draw all the attrs in subgraph;
                else treat all 1-dimen array as subgraphs and entries in each array as lines in each subgraph
        '''
        plt.figure()
        if chosen_labels is None:
            sub_dimension = self.dimen_size
            actual_dimension = 1
            for index, label in enumerate(self.attrlabels):
                plt.subplot(sub_dimension, 1, actual_dimension)
                plt.title(label)
                plt.plot(self.timelist, self.attrlists[index], label=label)
                plt.legend(loc="best")
                actual_dimension += 1
        else:
            sub_dimension = len(chosen_labels)
            actual_dimension = 1
            for chosen_label in chosen_labels:
                plt.subplot(sub_dimension, 1, actual_dimension)
                for label in chosen_label:
                    index = self.attrlabels.index(label)
                    plt.plot(self.timelist, self.attrlists[index], label=label)
                plt.legend(loc="best")
                plt.title(', '.join(chosen_label))
                actual_dimension += 1
        plt.xlabel('time/s')
        plt.show()

    def resample(self, resampled_freq: int, show: bool = False, inplace: bool = False):
        ''' resample series data with a new frequency, acomplished on the basis of scipy.signal.sample

        Args:
            resampled_freq: resampled frequency
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        origin_list = _self.attrlists
        origin_length = len(_self.attrlists)
        attr_length = _self.length
        origin_freq = _self.freq
        resampled_list = []
        for index in range(origin_length):
            origin_attr = origin_list[index]
            resampled_attr = signal.resample(origin_attr, int(attr_length/origin_freq*resampled_freq))
            resampled_list.append(resampled_attr)
        resampled_list = np.array(resampled_list)
        resampled_length = len(resampled_list[0])
        resampled_time = np.arange(_self.startts, _self.startts + 1 / resampled_freq * resampled_length, 1 / resampled_freq)
        if show == True:
            _self._show_resample(attr_length, origin_freq, resampled_freq, origin_list, resampled_list)
        _self.timelist = resampled_time
        _self.attrlists = resampled_list
        _self.freq = resampled_freq
        _self.length = resampled_length
        if not inplace:
            return _self

    def add_columns(self, column_names: list, values: list or np.ndarray = None, show: bool = False, inplace: bool = False):
        ''' add columns to STTimeseries object

        Args:
            column_name: list of column names
            values: values of columns
                if one-dimension list, new column will be single value
                if two-dimension list, new column will be a list
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        for index in range(len(column_names)):
            _type = type(values[index])
            if _type == list or _type == np.ndarray:
                if len(values[index]) == self.length:
                    _value = values[index]
                else:
                    raise Exception(f"length of values[{index}] mismatches with self length {_self.length}.")
            else:
                _value = np.array([values[index]] * self.length)
            _name = column_names[index]
            _self = _self._add_column(_name, _value)
        if show:
            _self.show()
        if not inplace:
            return _self

    def combine(self, series: object or list, show: bool = False, inplace: bool = False):
        ''' combine series data which have the same frequency

        Args:
            combined_series: a single or a list of STTimeseries object to be combined.
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        _type = type(series)
        if _type == list:
            _series = []
            for x in series:
                if type(x) == STTimeseries:
                    _series.append(x.copy())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            _self._combine_several(_series)
        elif _type == STTimeseries:
            _self._combine_one(series.copy())
        if show:
            _self.show()
        if not inplace:
            return _self

    def concat(self, series: object or list, show: bool = False, inplace: bool = False):
        ''' concat series data which have the same dimension size

        Args:
            combined_series: a single or a list of STTimeseries object to be concatted.
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        _type = type(series)
        if _type == list:
            _series = []
            for x in series:
                if type(x) == STTimeseries:
                    _series.append(x.copy())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            _self._concat_several(_series)
        elif _type == STTimeseries:
            _self._concat_one(series.copy())
        if show:
            _self.show()
        if not inplace:
            return _self

    def extract(self, attrs: list = None, show: bool = False, inplace: bool = False):
        ''' extract attrs from series data

        Args:
            attrs: names of chosen dimensions to be extracted
                if not provided, all columns will be extracted
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        templabels, templists = _self._handle_attrs(attrs)
        _self.attrlabels = templabels
        _self.attrlists = templists
        _self.dimen_size = len(templabels)
        if show:
            _self.show()
        if not inplace:
            return _self

    def cut(self, attrs: list = None, start: int = None, end: int = None, form: str = 'point', show: bool = False, inplace: bool = False):
        ''' cut timestamp in time dimension

        Args:
            attrs: names of columns to be cut
                if not provided, all columns will be cut
            start: start position of cut
                if not provided, cut from the very front position
            end: end position of cut
                if not provided, cut to the very last position
            form: default is "point", type of start and end
                if "point", start and end would mean absolute positions
                if "time", start and end would mean timestamp and need to multiply frequency to get the absolute positions
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        templabels, templists = _self._handle_attrs(attrs)
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
        timelist = _self.timelist.copy()[start:end]
        templists = np.array([attr[start:end] for attr in templists])
        _self.attrlists, _self.attrlabels, _self.timelist, \
            _self.startts, _self.length, _self.dimen_size = \
            templists, templabels, timelist, \
            timelist[0], len(timelist), len(templists)
        if show:
            _self.show()
        if not inplace:
            return _self

    def normalize(self, attrs: list = None, strategy: str = 'minmax', show: bool = False, inplace: bool = False):
        ''' normalize attributes by different strategies

        Args:
            attrs: names of columns to be normalized
                if not provided, all columns will be normalized
            strategy: normalize strategy
                minmax: normalize by minmax strategy to [-1, 1]
            show: if True, show the resampled signal with matplotlib.pyplot
            inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object

        Returns:
            None or a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        templabels, templists = _self._handle_attrs(attrs)
        attrlists = []
        for templist in templists:
            attrlist = _self._normalize_minmax(templist)
            attrlists.append(attrlist)
        _self.attrlists, _self.attrlabels, _self.dimen_size =\
            np.array(attrlists), templabels, len(templabels)
        if show:
            _self.show()
        if not inplace:
            return _self

    def savefile(self, name: str, path: str = None, attrs: list = None, annotation: str = None, savetime: bool = True, format: str = 'tensor'):
        ''' save current time series object as a tensor file, time column [if exists] shall always be stored as the first column

        Args:
            name: name of the file to be saved
            path: default is None, parent directory
            attr: default is None
                if assigned, only save required columns
            annotation: annotations which will be saved at the first line of the file
            savetime: default is True, save time dimension
            format: support 'tensor', 'csv' format

        Returns:
            None
        '''
        if path is None:
            path = f'./{name}.{format}'
        else:
            path = f'{path}{name}.{format}'
        templabels, templists = self._handle_attrs(attrs)
        if format == 'tensor':
            self._savefile_tensor(path, templists, templabels, annotation, savetime)
        elif format == 'csv':
            self._savefile_csv(path, templists, templabels, savetime)
        else:
            raise Exception(f'{format} not supported!')

    def _show_resample(self, attr_length: int, origin_freq: int, resampled_freq: int, origin_list: np.ndarray, resampled_list: np.ndarray):
        ''' draw resampled time series figure
        '''
        plt.figure()
        sub_dimension = len(resampled_list)
        actual_dimension = 1
        for label in self.attrlabels:
            x_origin = np.arange(0, attr_length/origin_freq, 1/origin_freq)
            x_resampled = np.arange(0, attr_length/origin_freq, 1/resampled_freq)
            plt.subplot(sub_dimension, 1, actual_dimension)
            index = self.attrlabels.index(label)
            plt.title(label)
            plt.plot(x_origin, origin_list[index], 'r-', label='origin')
            plt.plot(x_resampled, resampled_list[index], 'g.', label='resample')
            plt.legend(loc="best")
            actual_dimension += 1
        plt.xlabel('time/s')
        plt.show()

    def _add_column(self, _name: str, _value: np.ndarray):
        ''' add one column to STTimeseries object
        '''
        self.dimen_size += 1
        self.attrlabels.append(_name)
        self.attrlists = np.concatenate((self.attrlists, [_value]), axis=0)
        return self

    def _combine_one(self, obj: object):
        ''' combine single STTimeseries object
        '''
        if not self.freq == obj.freq:
            raise Exception(f'Frequency not matched, with {self.freq} and {obj.freq}')
        for label in obj.attrlabels:
            if label in self.attrlabels:
                for i in range(1, 10000):
                    if not (label + '_' + str(i)) in self.attrlabels:
                        self.attrlabels.extend([label + '_' + str(i)])
                        break
            else:
                self.attrlabels.extend([label])
        self.dimen_size += obj.dimen_size
        self.attrlists = np.concatenate([self.attrlists, obj.attrlists])

    def _combine_several(self, combined_series: list):
        ''' combine several STTimeseries object
        '''
        for obj in combined_series:
            self._combine_one(obj)

    def _concat_one(self, obj: object):
        ''' concat single STTimeseries object
        '''
        if not self.dimen_size == obj.dimen_size:
            raise Exception(f'dimension sizes are not the same with self {self.dimen_size} and obj {obj.dimen_size}')
        for i in range(len(self.attrlabels)):
            if not self.attrlabels[i] == obj.attrlabels[i]:
                raise Exception(f'{i}th dimension is not corresponding with self {self.attrlabels[i]} and obj {obj.attrlabels[i]}')
        self.attrlists = np.concatenate((self.attrlists, obj.attrlists), axis=1)
        self.length = len(self.attrlists[0])
        self.timelist = np.arange(self.startts, 1/self.freq*self.length, 1 / self.freq)

    def _concat_several(self, concated_series: list):
        ''' concat several STTimeseries object
        '''
        for obj in concated_series:
            self._concat_one(obj)

    def _normalize_minmax(self, attrlist: np.ndarray):
        ''' normalize by minmax strategy to [-1, 1]
        '''
        _min = np.min(attrlist)
        _max = np.max(attrlist)
        _middle = (_min+_max) / 2
        attrlist = (attrlist - _middle) / (_max - _min) * 2
        return attrlist

    def _savefile_tensor(self, path: str, templists: np.ndarray, templabels: list, annotation: str, time_flag: bool):
        ''' save file in tensor format
        '''
        templists = templists.T
        templists = [','.join(map(lambda x:str(x), t)) for t in templists]
        with open(path, 'w') as writer:
            if not annotation is None:
                writer.write(f'# {annotation} \n')
            writer.write(f'# time {" ".join(templabels)} \n')
            for i in range(self.length):
                if time_flag:
                    writer.write(f'{self.timelist[i]},{templists[i]}\n')
                else:
                    writer.write(f'{templists[i]}\n')

    def _savefile_csv(self, path: str, templists: np.ndarray, templabels: list, time_flag: bool):
        ''' save file in csv format
        '''
        if time_flag:
            timelist = np.array(self.timelist)
            templists = np.concatenate(([timelist], templists), axis=0)
            _ = ['time']
            _.extend(templabels)
            templabels = _
        templists = templists.T
        data_frame = pd.DataFrame(templists, columns=templabels)
        data_frame.to_csv(path, index=None)

    def _handle_inplace(self, inplace):
        ''' return a new object or origin object
        '''
        if inplace:
            _self = self
        else:
            _self = self.copy()
        return _self

    def _handle_attrs(self, attrs):
        ''' return attrlabels and attrlists
        '''
        if attrs is None:
            templabels = self.attrlabels
            templists = self.attrlists
        else:
            templabels = []
            templists = []
            for attr in attrs:
                if not attr in self.attrlabels:
                    raise Exception(f'Attr {attr} is not found')
                templabels.append(attr)
                index = self.attrlabels.index(attr)
                templists.append(self.attrlists[index])
            templists = np.array(templists)
        return templabels, templists

    def filter(self, order, criterion, mode, show=False, inplace=False):
        # TODO to be finished
        if type(criterion) == list and mode != 'bandpass' or type(criterion) == float and (mode != 'highpass' and mode != 'lowpass'):
            raise Exception('criterion not fit mode')
        b, a = signal.butter(order, criterion, mode)
        if show:
            plt.figure()
