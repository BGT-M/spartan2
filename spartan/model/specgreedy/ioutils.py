#!/usr/bin/env python3
# -*- coding=utf-8 -*-

# sys
import os
import sys
import argparse

# third-party libs
from scipy import io
import scipy.sparse as sps
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def append_suffix(fn, suffix):
    fn_out, fnext_out = os.path.splitext(os.path.basename(fn))
    return fn_out + suffix + fnext_out


def read_file(fn, mode='r'):
    if '.gz' == fn[-3:]:
        fn = fn[:-3]
    if os.path.isfile(fn):
        f = open(fn, mode)
    elif os.path.isfile(fn + '.gz'):
        import gzip
        f = gzip.open(fn + '.gz', mode)
    else:
        ValueError('File: {} or its zip file dose NOT exist'.format(fn))
        sys.exit(1)
    return f


def save_simple_dictdata(sim_dict, outfn, sep=':'):
    with open(outfn, 'w') as fp:
        for k, v in sim_dict.items():
            fp.writelines('{}{}{}\n'.format(k, sep, v))
        fp.close()


def load_simple_dictdata(infn, key_type=int, val_type=float, sep=':'):
    sim_dict = dict()
    with read_file(infn, 'r') as fp:
        for line in fp:
            tokens = line.strip().split(sep)
            sim_dict[key_type(tokens[0])] = val_type(tokens[1])
        fp.close()
    return sim_dict


def save_dictlist(dict_list, outfn, sep_dict=':', sep_list=','):
    with open(outfn, 'w') as fp:
        for k, ls in dict_list.items():
            if type(ls) is not list:
                ValueError('The value of the data is NOT list type!.')
                break
            ls_str = sep_list.join(str(t) for t in ls)
            fp.writelines('{}{}{}\n'.format(k, sep_dict, ls_str))
        fp.close()


def load_dictlist(infn, key_type=str, val_type=str, sep_dict=':', sep_list=','):
    dict_list = dict()
    with read_file(infn, 'r') as fp:
        for line in fp:
            tokens = line.strip().split(sep_dict)
            lst = [val_type(tok) for tok in tokens[1].strip().split(sep_list)]
            dict_list[key_type(tokens[0])] = lst
        fp.close()

    return dict_list


def save_simple_list(sim_list, outfn):
    with open(outfn, 'w') as fp:
        line_str = '\n'.join(str(t) for t in sim_list)
        fp.writelines(line_str)
        fp.close()


def load_simple_list(infn, dtype=None):
    sim_list = list()
    with read_file(infn, 'r') as fp:
        for line in fp:
            t = line.strip()
            if t == '':
                continue
            if dtype is not None:
                t = dtype(t)
            sim_list.append(t)
        fp.close()

    return sim_list


def load_mat(infn, var_name='data'):
    mat = dict(io.loadmat(infn, appendmat=True, squeeze_me=True))
    subs = mat[var_name]['subs'].tolist()
    vals = mat[var_name]['vals'].tolist()
    return subs, vals


def loadedge2sm(infile, mtype=sps.csc_matrix, weighted=False, dtype=int, delimiter=' ',
                idstartzero=True, issquared=False, comments='%'):
    '''
    load edge list into sparse matrix
    matrix dimensions are decided by max row id and max col id
    support csr, coo, csc matrix
    '''
    xs = []
    ys = []
    data = []

    if idstartzero is True:
        offset = 0
    else:
        offset = -1

    with read_file(infile, 'r') as fin:
        for line in fin:
            if line.startswith(comments):
                continue
            # print(line)
            coords = line.strip().split(delimiter)
            # print(coords)
            if issquared:
                if coords[0] == coords[1]:
                    continue
            xs.append(int(coords[0]) + offset)
            ys.append(int(coords[1]) + offset)
            if weighted:
                data.append(dtype(coords[2]))
            else:
                data.append(1)
        fin.close()

    m = max(xs) + 1
    n = max(ys) + 1

    if issquared is False:
        M, N = m, n
    else:
        M = N = max(m,n)

    sm = mtype((data, (xs, ys)), shape=(M, N))
    return sm, (m, n)


def savesm2edgelist(sm, ofile, idstartzero=True, delimiter=' ', kformat=None):
    '''
    Save sparse matrix into edgelist
    kformat = "{0:.1f}"
    '''
    offset = 0 if idstartzero else 1
    lsm = sm.tolil()
    with open(ofile, 'wb') as fout:
        i = 0
        for row, d in zip(lsm.rows, lsm.data):
            for j,w in zip(row, d):
                if kformat is not None:
                    w = str.format(kformat, w)
                ostr = delimiter.join([str(i+offset), str(j+offset), str(w)])
                #fout.write('{} {} {}\n'.format(i+offset, j+offset, w))
                fout.writelines(ostr + '\n')
            i+=1
        fout.close()
    return
