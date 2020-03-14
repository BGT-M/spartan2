import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
            sub_dimension = len(self.attrlists)
            actual_dimension = 1
            for label in self.attrlabels:
                plt.subplot(sub_dimension, 1, actual_dimension)
                index = self.attrlabels.index(label)
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
        origin_list = self.attrlists
        origin_length = len(self.attrlists)
        attr_length = self.length
        origin_freq = self.freq
        resampled_list = []
        for index in range(origin_length):
            origin_attr = origin_list[index]
            resampled_attr = signal.resample(origin_attr, int(attr_length/origin_freq*resampled_freq))
            resampled_list.append(resampled_attr)
        resampled_list = np.array(resampled_list)
        resampled_length = len(resampled_list[0])
        resampled_time = np.arange(self.startts, self.startts + 1 / resampled_freq * resampled_length, 1 / resampled_freq)
        if show == True:
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
            plt.show()
        if inplace == True:
            self.timelist = resampled_time
            self.attrlists = resampled_list
            self.freq = resampled_freq
            self.length = resampled_length
        else:
            return STTimeseries(resampled_time, resampled_list, self.attrlabels.copy(), resampled_freq, self.startts)

    def combine(self, combined_series, inplace=True):
        ''' combine series data which have the same frequency, can be a single STTimeseries object or a list of STTImeseries objects
            @param combined_series: series to be combined
            @param inplace:
                if True, update origin object's variable
                if False, return a new STTimeseries object
        '''
        if inplace:
            origin_series = self
        else:
            origin_series = self.copy()
        if type(combined_series) == list:
            _combined_series = []
            for x in combined_series:
                if type(x) == STTimeseries:
                    _combined_series.append(x.copy())
                else:
                    raise Exception(f'list contains non-STTimeseries object')
            self._combine_several(_combined_series, origin_series)
        elif type(combined_series) == STTimeseries:
            self._combine_one(combined_series.copy(), origin_series)
        if not inplace:
            return origin_series
        

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
        if form == 'point':
            start = start
            end = end
        elif form == 'time':
            if not start is None:
                start = start * self.freq
            if not end is None:
                end = end * self.freq
        else:
            raise Exception('Value of parameter form is not defined!')
        if start is None:
            start = 0
        if end is None:
            end = self.length
        timelist = self.timelist.copy()
        timelist = timelist[start:end]
        templists = [attr[start:end] for attr in templists]
        templists = np.array(templists)
        if attrs is None and inplace:
            self.attrlists = templists
            self.timelist = timelist
            self.startts = timelist[0]
            self.length = len(templists[0])
        else:
            return STTimeseries(timelist, templists, templabels)


    def filter(self, order, criterion, mode, show=False, inplace=False):
        if type(criterion) == list and mode != 'bandpass' or type(criterion) == float and (mode != 'highpass' and mode != 'lowpass'):
            raise Exception('criterion not fit mode')
        b, a = signal.butter(order, criterion, mode)
        if show:
            plt.figure()

    def savefile(self, name, path=None, attrs=None, annotation=None, time=True):
        ''' save current time series object as a tensor file, time column [if exists] shall always be stored as the first column
            @param name: name of the file to be saved
            @param path: default is None, parent directory
            @param attr: default is None
                if assigned, only save required columns
            @param annotation: annotations which will be saved at the first line of the file
            @param time: default is True, save time dimension
        '''
        if path is None:
            path = f'./{name}.tensor'
        else:
            path = f'{path}{name}.tensor'
        if time == False:
            time_flag = False
        elif time == True:
            time_flag = True
            timelist = self.timelist
        else:
            time_flag = True
            if time in self.attrlabels:
                timelist = self.attrlists[self.attrlabels.index(time)]
                np.delete(self.attrlists, self.attrlabels.index(time))
            else:
                raise Exception('time dimension assigned error')
        if attrs is None:
            templists = self.attrlists
            templabels = self.attrlabels
        else:
            templists = []
            templabels = attrs
            for attr in attrs:
                if not attr in self.attrlabels:
                    raise Exception(f'Attr {attr} not found!')
                index = self.attrlabels.index(attr)
                templists.append(self.attrlists[index])
            templists = np.array(templists)
        templists = templists.T
        templists = [','.join(map(lambda x:str(x), t)) for t in templists]
        with open(path, 'w') as writer:
            if not annotation is None:
                writer.write(f'# {annotation} \n')
            writer.write(f'# time {" ".join(templabels)} \n')
            for i in range(self.length):
                if time_flag:
                    writer.write(f'{timelist[i]},{templists[i]}\n')
                else:
                    writer.write(f'{templists[i]}\n')


    def _combine_one(self, combined_series, origin_series):
        if not origin_series.freq == combined_series.freq:
            raise Exception(f'Frequency not matched, with {self.freq} and {combined_series.freq}')
        for label in combined_series.attrlabels:
            if label in origin_series.attrlabels:
                for i in range(1, 10000):
                    if not (label + '_' + str(i)) in origin_series.attrlabels:
                        origin_series.attrlabels.extend([label + '_' + str(i)])
                        break
            else:
                origin_series.attrlabels.extend([label])
        origin_series.attrlists = np.concatenate([origin_series.attrlists, combined_series.attrlists])

    def _combine_several(self, combined_series, origin_series):
        for obj in combined_series:
            origin_series._combine_one(obj, origin_series)