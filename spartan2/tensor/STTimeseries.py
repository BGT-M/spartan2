import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200


class STTimeseries:
    def __init__(self, time, attrlists, attrlabels, freq=None, startts=None):
        ''' init STTimeseries object
            @param time: time dimension of data
            @param attrlists: signal dimensions of data
            @param attrlabels: labels of data, positions corresponding to data
            @params freq: frequency of the signal, default is None
                if time dimension is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, freq will not work and will be calculated by the time sequence
            @param startts: start timestamp, default is None
                if time is not provided, this parameter is needed to initiate time dimension
                if time dimension is provided, startts will not work and will be calculated by the time sequence
        '''
        self.length = len(attrlists[0])
        self.dimen_size = len(attrlists)
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

    def __len__(self):
        return self.length

    def copy(self):
        time = self.timelist.copy()
        attrlists = self.attrlists.copy()
        attrlabels = self.attrlabels.copy()
        return STTimeseries(time, attrlists, attrlabels)

    def show(self, chosen_labels=None):
        ''' draw series data with matplotlib.pyplot
            @type chosen_labels: [[]]
            @param chosen_labels:
                if None, draw all the attrs in subgraph;
                or treat all 1-dimen array as subgraphs and entries in each array as lines in each subgraph
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
                plt.title(chosen_label)
                actual_dimension += 1
        plt.show()

    def resample(self, resampled_freq, show=False, inplace=False):
        ''' resample series data with a new frequency, acomplish on the basis of scipy.signal.sample
            @param resampled_freq: resampled frequency
            @param show: if True, show the resampled signal with matplotlib.pyplot
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
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
            plt.figure()
            sub_dimension = len(resampled_list)
            actual_dimension = 1
            for label in _self.attrlabels:
                x_origin = np.arange(0, attr_length/origin_freq, 1/origin_freq)
                x_resampled = np.arange(0, attr_length/origin_freq, 1/resampled_freq)
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = _self.attrlabels.index(label)
                plt.title(label)
                plt.plot(x_origin, origin_list[index], 'r-', label='origin')
                plt.plot(x_resampled, resampled_list[index], 'g.', label='resample')
                plt.legend(loc="best")
                actual_dimension += 1
            plt.show()
        _self.timelist = resampled_time
        _self.attrlists = resampled_list
        _self.freq = resampled_freq
        _self.length = resampled_length
        if not inplace:
            return _self

    def add_column(self, column_name, value, inplace=False):
        _self = self._handle_inplace(inplace)
        _self.attrlabels.extend([column_name])
        _self.dimen_size += 1
        attrlist = np.array([value] * _self.length)
        _self.attrlists = np.concatenate((_self.attrlists, [attrlist]), axis=0)
        if not inplace:
            return _self

    def combine(self, series, inplace=False):
        ''' combine series data which have the same frequency, can be a single STTimeseries object or a list of STTImeseries objects
            @param combined_series: series to be combined
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        if type(series) == list:
            _series = []
            for x in series:
                if type(x) == STTimeseries:
                    _series.append(x.copy())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            self._combine_several(_series, _self)
        elif type(series) == STTimeseries:
            self._combine_one(series.copy(), _self)
        if not inplace:
            return _self

    def concat(self, series, inplace=False):
        _self = self._handle_inplace(inplace)
        if type(series) == list:
            _series = []
            for x in series:
                if type(x) == STTimeseries:
                    _series.append(x.copy())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            self._concat_several(_series, _self)
        elif type(series) == STTimeseries:
            self._concat_one(series.copy(), _self)
        if not inplace:
            return _self

    def extract(self, attrs=None, inplace=False):
        ''' extract attrs from series data
            @param attrs: default is None, return each dimension
                if not None, return required dimensions
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        _attrlists = list(_self.attrlists)
        for attr in _self.attrlabels.copy():
            if attr not in attrs:
                index = _self.attrlabels.index(attr)
                del(_attrlists[index])
                _self.attrlabels.remove(attr)
        _self.attrlists = np.array(_attrlists)
        if not inplace:
            return _self

    def cut(self, attrs=None, start=None, end=None, form='point', inplace=False):
        ''' cut columns in time dimension
            @type attrs: array
            @param attrs: default is None, columns to be cut
                if not None, attr sprcified in attrs will be cut AND param inplace will be invalid
            @param start: default is None, start position
                if start is None, cut from the very front position
            @param end: default is None, end position
                if end is None, cut to the very last position
            @param form: default is point, type of start and end
                if "point", start and end would mean absolute positions of columns
                if "time", start and end would mean timestamp and need to multiply frequenct to get the absolute positions
            @param inplace: default if False, IF attrs is not None, this param will be invalid
                if False, function will return a new STTimeseiries object
                if True, function will make changes in current STTimeseries object
        '''
        _self = self._handle_inplace(inplace)
        templabels, templists = self._handle_attrs(attrs)
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
        _self.attrlists = templists
        _self.timelist = timelist
        _self.startts = timelist[0]
        _self.length = len(templists[0])
        if not inplace:
            return _self

    def filter(self, order, criterion, mode, show=False, inplace=False):
        # TODO to be finished
        if type(criterion) == list and mode != 'bandpass' or type(criterion) == float and (mode != 'highpass' and mode != 'lowpass'):
            raise Exception('criterion not fit mode')
        b, a = signal.butter(order, criterion, mode)
        if show:
            plt.figure()

    def normalize(self, attrs=None, inplace=False):
        _self = self._handle_inplace(inplace)
        if attrs is None:
            attrs = _self.attrlabels
        _attrlists = list(_self.attrlists)
        for i, value in enumerate(_self.attrlabels):
            if value in attrs:
                _attrlists[i] = self._normalize(_attrlists[i])
        _self.attrlists = np.array(_attrlists)
        if not inplace:
            return _self

    def savefile(self, name, path=None, attrs=None, annotation=None, time=True, format='tensor'):
        ''' save current time series object as a tensor file, time column [if exists] shall always be stored as the first column
            @param name: name of the file to be saved
            @param path: default is None, parent directory
            @param attr: default is None
                if assigned, only save required columns
            @param annotation: annotations which will be saved at the first line of the file
            @param time: default is True, save time dimension
        '''
        if path is None:
            path = f'./{name}.{format}'
        else:
            path = f'{path}{name}.{format}'
        if time == False:
            time_flag = False
        elif time == True:
            time_flag = True
        else:
            time_flag = True
            if time in self.attrlabels:
                _pos = self.attrlabels.index(time)
                self.timelist = self.attrlists[_pos]
                _attrs = list(self.attrlists)
                del(_attrs[_pos])
                self.attrlists = np.array(_attrs)
                self.attrlabels.remove(time)
            else:
                raise Exception('time dimension assigned error')
        templabels, templists = self._handle_attrs(attrs)
        if format == 'tensor':
            self._savefile_tensor(path, templists, templabels, annotation, time_flag)
        elif format == 'csv':
            self._savefile_csv(path, templists, templabels, time_flag)
        else:
            raise Exception(f'{format} not supported!')

    def _combine_one(self, obj, _self):
        if not _self.freq == obj.freq:
            raise Exception(f'Frequency not matched, with {_self.freq} and {obj.freq}')
        for label in obj.attrlabels:
            if label in _self.attrlabels:
                for i in range(1, 10000):
                    if not (label + '_' + str(i)) in _self.attrlabels:
                        _self.attrlabels.extend([label + '_' + str(i)])
                        break
            else:
                _self.attrlabels.extend([label])
        _self.attrlists = np.concatenate([_self.attrlists, obj.attrlists])

    def _combine_several(self, combined_series, _self):
        for obj in combined_series:
            _self._combine_one(obj, _self)

    def _concat_one(self, obj, _self):
        if not _self.dimen_size == obj.dimen_size:
            raise Exception(f'dimension sizes are not the same with self {_self.dimen_size} and obj {obj.dimen_size}')
        for i in range(len(_self.attrlabels)):
            if not _self.attrlabels[i] == obj.attrlabels[i]:
                raise Exception(f'{i}th dimension is not corresponding with self {_self.attrlabels[i]} and obj {obj.attrlabels[i]}')
        _self.attrlists = np.concatenate((_self.attrlists, obj.attrlists), axis=1)
        _self.length = len(_self.attrlists[0])
        _self.timelist = np.arange(_self.startts, 1/_self.freq*_self.length, 1 / _self.freq)

    def _concat_several(self, concated_series, _self):
        for obj in concated_series:
            _self._concat_one(obj, _self)

    def _savefile_tensor(self, path, templists, templabels, annotation, time_flag):
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

    def _savefile_csv(self, path, templists, templabels, time_flag):
        if time_flag:
            timelist = np.array(self.timelist)
            templists = np.concatenate(([timelist], templists), axis=0)
            _ = ['time']
            _.extend(templabels)
            templabels = _
        templists = templists.T
        data_frame = pd.DataFrame(templists, columns=templabels)
        data_frame.to_csv(path, index=None)

    def _normalize(self, attrlist):
        _min = np.min(attrlist)
        _max = np.max(attrlist)
        _middle = (_min+_max) / 2
        attrlist = (attrlist - _middle) / (_max - _min) * 2
        return attrlist

    def _handle_inplace(self, inplace):
        if inplace:
            _self = self
        else:
            _self = self.copy()
        return _self

    def _handle_attrs(self, attrs):
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
