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
from . import TensorData
from .basicutil import set_trace


class File():

    def __init__(self, filename, mode, idxtypes):
        self.filename = filename
        self.mode = mode
        self.idxtypes = idxtypes
        self.dtypes = None

    def get_sep_of_file(self):
        '''
        return the separator of the line.
        :param infn: input file
        '''
        sep = None
        with self._open() as fp:
            for line in fp:
                line = line.decode(
                    'utf-8') if isinstance(line, bytes) else line
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


class TensorFile(File):
    def _open(self):
        if 'r' not in self.mode:
            self.mode += 'r'
        f = open(self.filename, self.mode)
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
                for i, tp in self.idxtypes:
                    try:
                        tline.append(tp(coords[i]))
                    except Exception:
                        raise Exception(
                            f"The {i}-th col does not match the given type {tp} in line:\n{line}")
                tensorlist.append(tline)
        tensorlist = pd.DataFrame(tensorlist)
        return tensorlist


class CSVFile(File):
    def _open(self):
        f = pd.read_csv(self.filename, header=None)
        column_names = f.columns
        self.dtypes = {}
        if not self.idxtypes is None:
            for idx, typex in self.idxtypes:
                self.dtypes[column_names[idx]] = self.transfer_type(typex)
            f = pd.read_csv(self.filename, dtype=self.dtypes, header=None)
        else:
            f = pd.read_csv(self.filename, header=None)
        return f

    def _read(self):
        tensorlist = pd.DataFrame()
        _file = self._open()
        if not self.idxtypes is None:
            idx = [i[0] for i in self.idxtypes]
            tensorlist = _file[idx]
        else:
            tensorlist = _file
        return tensorlist

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


def _read_data(filename: str, idxtypes: list) -> object:
    """Check format of file and read data from file.

    Default format is .tensor. Now we support read from csv, gz, tensor.

    Parameters
    ----------
    filename : str
        file name
    idxtypes : list
        type of columns

    Returns
    ----------
    Data object read from file

    Raises
    ----------
    Exception
        if file cannot be read, raise a FileNotFoundError.
    """

    _class = None
    _postfix = os.path.splitext(filename)[-1]
    if _postfix == ".csv":
        _filename = filename
        _class = CSVFile
    elif _postfix == ".tensor":
        _filename = filename
        _class = TensorFile
    elif _postfix in ['.gz', '.bz2', '.zip', '.xz']:
        _filename = filename
        _class = CSVFile
    else:
        raise FileNotFoundError(
            f"Error: Can not find file {filename}, please check the file path!\n")
    _obj = _class(_filename, 'r', idxtypes)
    _data = _obj._read()
    return _data


def _check_compress_file(path: str, cformat=['.gz', '.bz2', '.zip', '.xz']):
    valpath = None
    if os.path.isfile(path):
        valpath = path
    else:
        for cf in cformat:
            if os.path.isfile(path+cf):
                valpath = path + cf
                return valpath
    if not valpath is None:
        return valpath
    else:
        raise FileNotFoundError(f"{path} cannot be found.")


def _aggregate(data_list):
    if len(data_list) < 1:
        raise Exception("Empty list of data")
    elif len(data_list) == 1:
        return data_list[0]
    else:
        pass


def loadTensor(path: str, col_idx: list = None, col_types: list = None):
    if path is None:
        raise FileNotFoundError('Path is missing.')
    path = _check_compress_file(path)
    import glob
    files = glob.glob(path)
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

    data_list = []
    for _file in files:
        data_list.append(_read_data(_file, idxtypes))
    data = _aggregate(data_list)
    return TensorData(data)
