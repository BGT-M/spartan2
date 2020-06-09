import os
import sys
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class File(metaclass=ABCMeta):
    def __init__(self, name, mode, idxtypes):
        self.name = name
        self.mode = mode
        self.idxtypes = idxtypes

    def get_sep_of_file(self):
        '''
        return the separator of the line.
        :param infn: input file
        '''
        sep = None
        with self._open() as fp:
            for line in fp:
                if (line.startswith("%") or line.startswith("#")):
                    continue
                line = line.strip()
                if (" " in line):
                    sep = " "
                if ("," in line):
                    sep = ","
                if (";" in line):
                    sep = ';'
                if ("\t" in line):
                    sep = "\t"
                break
        self.sep = sep

    @abstractmethod
    def _open(self):
        pass

    @abstractmethod
    def _read(self):
        pass


class GZFile(File):
    def _open(self):
        import gzip
        if 'r' not in self.mode:
            self.mode += 'r'
        if 'b' not in self.mode:
            self.mode += 'b'
        f = gzip.open(self.name, self.mode)
        return f

    def _read(self):
        tensorlist = []
        self.get_sep_of_file()
        with self._open() as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("#"):
                    continue
                coords = line.split(self.sep)
                tline = []
                try:
                    for i, tp in self.idxtypes:
                        tline.append(tp(coords[i]))
                except Exception as e:
                    raise Exception(f"The {i}-th col does not match the given type {tp} in line:\n{line}")
                tensorlist.append(tline)
        return tensorlist


class TensorFile(File):
    def _open(self):
        if 'r' not in self.mode:
            self.mode += 'r'
        f = open(self.name, self.mode)
        return f

    def _read(self):
        tensorlist = []
        self.get_sep_of_file()
        with self._open() as fin:
            for line in fin:
                line = line.strip()
                if line.startswith("#"):
                    continue
                coords = line.split(self.sep)
                tline = []
                try:
                    for i, tp in self.idxtypes:
                        tline.append(tp(coords[i]))
                except Exception as e:
                    raise Exception(f"The {i}-th col does not match the given type {tp} in line:\n{line}")
                tensorlist.append(tline)
        return tensorlist


class CSVFile(File):
    def _open(self):
        f = pd.read_csv(self.name)
        column_names = f.columns
        dtypes = {}
        if not self.idxtypes is None:
            for idx, typex in self.idxtypes:
                dtypes[column_names[idx]] = self.transfer_type(typex)
            f = pd.read_csv(self.name, dtype=dtypes)
        else:
            f = pd.read_csv(self.name)
        return f

    def _read(self):
        tensorlist = []
        _file = self._open()
        _names = []
        if not self.idxtypes is None:
            idx = [i[0] for i in self.idxtypes]
            for _id in idx:
                tensorlist.append(np.array(_file.iloc[:, _id].T))
                _names.append(_file.columns[_id])
            tensorlist = np.array(tensorlist).T
        else:
            tensorlist = np.array(_file)
            _names = _file.columns
        return tensorlist, _names

    def transfer_type(self, typex):
        if typex == float:
            _typex = 'float'
        elif typex == int:
            _typex = 'int'
        elif typex == str:
            _typex = 'object'
        else:
            _typex = 'object'
        return _typex


def checkfileformat(name, idxtypes):
    _class = None
    if os.path.isfile(name):
        _name = name
        _class = TensorFile
    elif os.path.isfile(name+'.tensor'):
        _name = name + '.tensor'
        _class = TensorFile
    elif os.path.isfile(name+'.gz'):
        _name = name + '.gz'
        _class = GZFile
    elif os.path.isfile(name+'.csv'):
        _name = name + '.csv'
        _class = CSVFile
    else:
        raise Exception(f"Error: Can not find file {name}, please check the file path!\n")
    _obj = _class(_name, 'r', idxtypes)
    _data = _obj._read()
    return _data
