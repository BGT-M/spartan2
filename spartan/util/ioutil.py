#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ioutil.py
@Desc    :   Input and output data function.
'''

# here put the import lib

import gzip
import os
import sys
import pandas as pd
import numpy as np
from . import TensorData
import csv
from .basicutil import set_trace


class File():

    def __init__(self, filename, mode, idxtypes):
        self.filename = filename
        self.mode = mode
        self.idxtypes = idxtypes
        self.dtypes = None

    @staticmethod
    def transfer_type(typex):
        if typex == float:
            _typex = 'float'
        elif typex == int:
            _typex = 'int'
        elif typex == str:
            _typex = 'object'
        else:
            _typex = 'object'
        return _typex

    def _open(self, **kwargs):
        pass

    def _read(self, **kwargs):
        pass


class GZipFile(File):
    def _open(self, **kwargs):
        f = gzip.open(self.filename, self.mode)
        pos = 0
        cur_line = f.readline()
        while cur_line.startswith(b"#"):
            pos = f.tell()
            cur_line = f.readline()
        f.seek(pos)
        _f = open(self.filename, self.mode)
        _f.seek(pos)
        fin = pd.read_csv(f, **kwargs)
        column_names = fin.columns
        self.dtypes = {}
        if self.idxtypes is not None:
            for idx, typex in self.idxtypes:
                self.dtypes[column_names[idx]] = self.transfer_type(typex)
            fin = pd.read_csv(_f, dtype=self.dtypes, **kwargs)
        else:
            fin = pd.read_csv(_f, **kwargs)
        return fin


class TensorFile(File):
    def _open(self, **kwargs):
        if 'r' not in self.mode:
            self.mode += 'r'
        f = open(self.filename, self.mode)
        pos = 0
        cur_line = f.readline()
        while cur_line.startswith('#'):
            pos = f.tell()
            cur_line = f.readline()
        f.seek(pos)
        _f = open(self.filename, self.mode)
        _f.seek(pos)
        fin = pd.read_csv(f, **kwargs)
        column_names = fin.columns
        self.dtypes = {}
        if not self.idxtypes is None:
            for idx, typex in self.idxtypes:
                self.dtypes[column_names[idx]] = self.transfer_type(typex)
            fin = pd.read_csv(_f, dtype=self.dtypes, **kwargs)
        else:
            fin = pd.read_csv(_f, **kwargs)
        return fin

    def _read(self, **kwargs):
        tensorlist = []
        _file = self._open(**kwargs)
        if not self.idxtypes is None:
            idx = [i[0] for i in self.idxtypes]
            tensorlist = _file[idx]
        else:
            tensorlist = _file
        return tensorlist


class CSVFile(File):
    def _open(self, **kwargs):
        f = pd.read_csv(self.filename, **kwargs)
        column_names = list(f.columns)
        self.dtypes = {}
        if not self.idxtypes is None:
            for idx, typex in self.idxtypes:
                self.dtypes[column_names[idx]] = self.transfer_type(typex)
            f = pd.read_csv(self.filename, dtype=self.dtypes, **kwargs)
        else:
            f = pd.read_csv(self.filename, **kwargs)
        return f

    def _read(self, **kwargs):
        tensorlist = pd.DataFrame()
        _file = self._open(**kwargs)
        if not self.idxtypes is None:
            idx = list(self.dtypes.keys())
            tensorlist = _file[idx]
        else:
            tensorlist = _file
        return tensorlist


class NPFile(File):
    def _open(self, **kwargs):
        f = np.load(self.filename)
        return f

    def _read(self, **kwargs):
        f = self._open(**kwargs)
        if self.filename.endswith(".npy"):
            df = pd.DataFrame(f)
        elif self.filename.endswith(".npz"):
            df = pd.DataFrame()
            for k in f.keys():
                df = pd.concat([df, pd.DataFrame(f[k])])
        else:
            df = None
        if not self.idxtypes is None:
            idx = [i[0] for i in self.idxtypes]
            types = [i[1] for i in self.idxtypes]
            df = df[idx]
            for i, _type in enumerate(types):
                df[i] = df[i].astype(self.transfer_type(_type))
        else:
            df = df
        return df


def _read_data(filename: str, idxtypes: list, **kwargs) -> object:
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
    mode = 'r'
    if _postfix == ".csv":
        _class = CSVFile
    elif _postfix == ".tensor":
        _class = TensorFile
    elif _postfix in [".npy", ".npz"]:
        _class = NPFile
    elif _postfix in [".gz", ".bz2", ".zip", ".xz"]:
        _class = CSVFile
    else:
        for _postfix in [".csv", ".tensor", ".npy", ".npz", ".gz", ".bz2", ".zip", ".xz"]:
            if os.path.isfile(filename + _postfix):
                _filename = filename + _postfix
                return _read_data(_filename, idxtypes, **kwargs)
        _class = CSVFile
    _obj = _class(filename, mode, idxtypes)
    _data = _obj._read(**kwargs)
    return _data


def _check_compress_file(path: str, cformat=[".gz", ".bz2", ".zip", ".xz"]):
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
        return pd.concat(data_list, axis=0, ignore_index=True)


def _isgzfile(filename):
    return filename.endswith(".gz")


"gzip file must be read and write in binary/bytes"


def _myopenfile(fnm, mode):
    f = None
    if 'w' in mode:
        if _isgzfile(fnm):
            import gzip
            mode = mode+'b' if 'b' != mode[-1] else mode
            f = gzip.open(fnm, mode)
        else:
            f = open(fnm, mode)
    else:
        if 'r' not in mode and 'a' not in mode:
            mode = 'r' + mode
        if os.path.isfile(fnm):
            if not _isgzfile(fnm):
                f = open(fnm, mode)
            else:
                import gzip
                mode = 'rb'
                f = gzip.open(fnm, mode)
        elif os.path.isfile(fnm+'.gz'):
            'file @fnm does not exists, use fnm.gz instead'
            print(
                '==file {} does not exists, read {}.gz instead'.format(fnm,
                                                                       fnm))
            import gzip
            mode = 'rb'
            f = gzip.open(fnm+'.gz', mode)
        else:
            print('file: {} or its zip file does NOT exist'.format(fnm))
            sys.exit(1)
    return f


def loadTensor(path: str,  col_idx: list = None, col_types: list = None, sep: str = ',', **kwargs):
    '''
    Parameters
    ------
        path:
            file-like object
    '''
    if not "header" in kwargs.keys():
        kwargs["header"] = None
    if path is None:
        raise FileNotFoundError('Path is missing.')
    if not "sep" in kwargs.keys():
        kwargs["sep"] = sep

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
            idxtypes = list(zip(col_idx, col_types))
        else:
            raise Exception(f"Error: input same size of col_types and col_idx")

    data_list = []
    for _file in files:
        _file = _check_compress_file(_file)
        data_list.append(_read_data(_file, idxtypes, **kwargs))
    data = _aggregate(data_list)
    return TensorData(data)


def loadTensorStream(filename: str, col_idx: list = None, col_types: list = None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist!")
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
    postfix = os.path.splitext(filename)[-1]
    if postfix == ".csv":
        f = open(filename, 'r')
        has_header = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)
        if has_header:
            next(f)
    elif postfix == ".tensor":
        f = open(filename, 'r')
    elif postfix in ['.gz', '.bz2', '.zip', '.xz']:
        f = open(filename, 'r')
        has_header = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)
        if has_header:
            next(f)
    else:
        raise FileNotFoundError(
            f"Error: Can not find file {filename}, please check the file!\n")
    return f, idxtypes


def loadFile2Dict(infn: str, n_keyelems: int = 1, key_elem_type: type = int,
                  value_elem_type: type = float, comments: str = "#", delimiter: str = ","):
    '''DESIGNED for EAGLEMINE: load dict data from the input file
    Parameters:
    -------
    :param infn: str
        Input data file
    :param n_keyelems: int
        The number of elements as the key of the 'dict' data, which should be [1, \max{\#elems} -  1]
        Default is 1.
    :param key_elem_type: type
        The type of the key element(s)
        Default is int.
    :param value_elem_type: type
        The type of the value element(s)
        Default is float.
    :param comments: str
        The comments (start character) of inputs.
        Default is "#".
    :param delimiter: str
        The separator of items in each line of inputs.
        Default is ",".
    '''
    data = dict()
    with open(infn, 'r') as fp:
        for line in fp:
            if line.startswith(comments):
                continue
            toks = line.strip().split(delimiter)
            n_elems = len(toks)
            if n_elems <= 1:
                raise ValueError(
                    "Invalid data in input file, which should contain two elements at least")
            if n_keyelems > 1:
                key = tuple(map(key_elem_type, toks[:n_keyelems]))
                if n_elems - n_keyelems > 1:
                    val = tuple(map(value_elem_type, toks[n_keyelems:]))
                else:
                    val = value_elem_type(toks[-1])
                data[key] = val
            elif n_keyelems == 1:
                if n_elems - n_keyelems > 1:
                    val = tuple(map(value_elem_type, toks[1:]))
                else:
                    val = value_elem_type(toks[1])
                data[key_elem_type(toks[0])] = val
            else:
                raise ValueError(
                    "Invalid parameter 'n_keyelems', which should be [{}, {}]".format(1, n_elems - 1))
        fp.close()
    return data


def loadHistogram(infn: str, comments: str = '#', delimiter: str = ','):
    '''Load histogram data
    Parameters:
    --------
    :param infn: str
        The input histogram file with following format:
         the 1st three lines started with ‘#’ are comments for some basic information, that is, 1st line shows the shape of 2-dimensional histogram;
         the 2nd line gives the corresponding real coordinate of each cell coordinate x, and
         the 3rd line is the corresponding real coordinate of each cell coordinate y, these two lines are used for visualizing the histogram.
         Then the followed lines are non-zero data records of the histogram.
    :param comments: str
            The comments (start character) of inputs.
            Default is "#".
        :param delimiter: str
            The separator of items in each line of inputs.
            Default is ",".
    '''
    shape, ticks_dims = list(), list()
    hist_arr = list()

    with open(infn, 'r') as fp:
        line_1st = fp.readline().strip()
        if not line_1st.startswith(comments):
            raise IOError(
                "Invalid input histogram file, which should start with {}".format(comments))
        shape = list(map(int, line_1st[1:].split(delimiter)))

        for m in range(len(shape)):
            line_m = fp.readline().strip()
            if not line_m.startswith(comments):
                raise IOError(
                    "Invalid input histogram file, which should start with {}".format(comments))
            ticks_m = list(map(float, line_m[1:].split(delimiter)))
            ticks_dims.append(ticks_m)

        for line in fp.readlines():
            if line.startswith(comments):
                continue
            toks = line.strip().split(delimiter)
            hist_arr.append(list(map(int, toks)))
        fp.close()

    return np.array(shape, int), ticks_dims, np.array(hist_arr, int)


def saveDictListData(dictls, outdata, delim=':', mode='w'):
    if _isgzfile(outdata):
        'possible mode is ab'
        mode = mode + 'b' if mode[-1] != 'b' else mode
    # write bytes
    ib = True if 'b' in mode else False

    with _myopenfile(outdata, mode) as fw:
        i = 0
        for k, l in dictls.items():
            if not isinstance(l, (list, np.ndarray)):
                print("This is not a dict of value list.", type(l))
                break
            if len(l) < 1:
                continue
            k = k.decode() if isinstance(k, bytes) else str(k)
            ostr = "{}{}".format(k, delim)
            if len(l) < 1:
                i += 1
                continue
            l = [x.decode() if isinstance(x, bytes) else str(x) for x in l]
            ostr = ostr + " ".join(l) + '\n'
            fw.write(ostr.encode() if ib else ostr)
        fw.close()
        if i > 0:
            print("Warn: total {} empty dict lists are removed".format(str(i)))


def loadDictListData(indata, ktype=str, vtype=str, delim=':', mode='r'):
    if _isgzfile(indata):
        mode = 'rb'
    # bytes
    ib = True if 'b' in mode else False

    dictls = {}
    with _myopenfile(indata, mode) as fr:
        for line in fr:
            line = line.decode() if ib else line
            line = line.strip().split(delim)
            lst = []
            for e in line[1].strip().split(' '):
                if vtype == int:
                    lst.append(vtype(float(e)))
                else:
                    lst.append(vtype(e))
            dictls[ktype(line[0].strip())] = lst
        fr.close()
    return dictls
