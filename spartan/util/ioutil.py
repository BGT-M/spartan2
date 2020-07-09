#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ioutil.py
@Desc    :   Input and output data function.
'''

# here put the import lib

import os
import sys
import pandas as pd
import numpy as np
from . import STTensor


class File():

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

    def _open(self):
        pass

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
                except Exception:
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
                except Exception:
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


def read_data(name: str, idxtypes: list) -> object:
    """Check format of file and read data from file.

    Default format is .tensor. Now we support read from csv, gz, tensor.

    Parameters
    ----------
    name : str
        file name
    idxtypes : list
        type of columns

    Returns
    ----------
    Data object read from file

    Raises
    ----------
    Exception
        if file cannot be read, raise an exception.
    """
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


def loadTensor(name: str, path: str, col_idx: list = None, col_types: list = None, hasvalue: bool = True, value_idx: int = None) -> STTensor:
    '''Interface of loadtensor function, read data from file.

    Parameters
    ----------
    name : str
        file name
    path : str
        file path
    col_idx : list
        id of chosen columns in data file
    col_types : list
        data type of each chosen column
    hasvalue : bool
        if time series data, this refers to whether time column exists
        if graph data, this refers to 
    value_idx : int
        id of value column

    Returns
    ----------
    STTensor object constructed from data

    Raises
    ----------
    Exception
        when shape of col_types mismatches col_idx, raise IOError
    '''
    if path is None:
        path = "inputData/"
    full_path = os.path.join(path, name)
    if col_types is None:
        if col_idx is None:
            idxtypes = None
        else:
            idxtypes = [(x, str) for x in col_idx]
    else:
        if col_idx is None:
            col_idx = [i for i in range(len(col_types))]
        if len(col_idx) == len(col_types):
            idxtypes = [(x, col_types[i]) for i, x in enumerate(col_idx)]
        else:
            raise Exception(f"Error: input same size of col_types and col_idx")
    if hasvalue and value_idx is None:
        value_idx = 0
    retval = read_data(full_path, idxtypes)
    names = None
    if type(retval) == tuple:
        tensorlist, names = retval
    else:
        tensorlist = retval
    tensor = STTensor(tensorlist, hasvalue, value_idx, names)
    print(tensor)
    return tensor
