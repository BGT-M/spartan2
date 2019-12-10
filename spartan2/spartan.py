#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import sys
import os
from . import system
import importlib
import sqlite3
import scipy.sparse.linalg as slin
from .ioutil import checkfilegz, get_sep_of_file, myreadfile
from .sttensor import STTensor
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix

# engine
engine = system.Engine()

# model
anomaly_detection = system.AnomalyDetection()
decomposition = system.Decomposition()
traingle_count = system.TraingleCount()

'''Input tensor format:
    format: att1, att2, ..., value1, value2, ...
    comment line started with #
    e.g.
    user obj 1
    ... ...
    if multivariate time series, hasvalue equals to the number of
    time series
    return: tensorlist
'''


def loadTensor(name, path, col_types=[int, int, int],
               hasvalue=1, col_idx=[]):

    if path == None:
        path = "inputData/"
    full_path = os.path.join(path, name)
    tensor_file = checkfilegz(full_path + '.tensor')

    if tensor_file is None:
        raise Exception(f"Error: Can not find file {tensor_file}[.gz], please check the file path!\n")

    # NOTE: zip and range are different in py3
    col_idx = [i for i in range(len(col_types))] if len(col_idx) == 0 else col_idx

    if len(col_idx) == len(col_types):
        idxtypes = [(col_idx[i], col_types[i]) for i in range(len(col_idx))]
    else:
        raise Exception(f"Error: input same size of col_types and col_idx")

    #import ipdb;ipdb.set_trace()
    sep = get_sep_of_file(tensor_file)
    tensorlist = []
    with myreadfile(tensor_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            coords = line.split(sep)
            tline = []
            try:
                for i, tp in idxtypes:
                    tline.append(tp(coords[i]))
            except Exception as e:
                raise Exception(f"The {i}-th col does not match the given type {tp} in line:\n{line}")
            tensorlist.append(tline)
    printTensorInfo(tensorlist, hasvalue)

    return STTensor(tensorlist, hasvalue)
    
#    for i in range(len(col_types)):
#        col_types[i] = convert_to_db_type(col_types[i])
#


def printTensorInfo(tensorlist, hasvalue):
    m = len(tensorlist[0]) - hasvalue
    print(f"Info: Tensor is loaded\n\
           ----------------------\n\
             attr     |\t{m}\n\
             values   |\t{hasvalue}\n\
             nonzeros |\t{len(tensorlist)}\n")


def config(frame_name):
    global ad_policy, tc_policy, ed_policy
    frame = importlib.import_module(frame_name)

    # algorithm list
    ad_policy = frame.AnomalyDetection()
    tc_policy = frame.TriangleCount()
    ed_policy = frame.Decomposition()


def bidegree(edgelist):
    sm = _get_sparse_matrix(edgelist, squared=True)

    sm_csr = sm.tocsr(copy=False)
    sm_csc = sm.tocsc(copy=False)

    # calculate degree
    Du = sm_csr.sum(axis=1).getA1()
    Dv = sm_csc.sum(axis=0).getA1()

    return Du, Dv


'''
def degree(edgelist):
    sm = _get_sparse_matrix(edgelist, True)

    sm_csr = sm.tocsr(copy = False)
    sm_csc = sm.tocsc(copy = False)

    # calculate degree
    Du = sm_csr.sum(axis = 1).getA1()
    Dv = sm_csc.sum(axis = 0).getA1()
    D = Du + Dv

    return D
'''


def _get_sparse_matrix(edgelist, squared=False):
    edges = edgelist[2]
    edge_num = len(edges)

    # construct the sparse matrix
    xs = [edges[i][0] for i in range(edge_num)]
    ys = [edges[i][1] for i in range(edge_num)]
    data = [1] * edge_num

    row_num = max(xs) + 1
    col_num = max(ys) + 1

    if squared:
        row_num = max(row_num, col_num)
        col_num = row_num

    sm = coo_matrix((data, (xs, ys)), shape=(row_num, col_num))

    return sm


def subgraph(edgelist, uid_array, oid_array=None):
    if oid_array == None:
        squared = True
    else:
        squared = False

    # create db connection
    con = sqlite3.connect(":memory:")
    cur = con.cursor()

    # create edge table
    sql_str = '''CREATE TABLE EDGE
                    (id INTEGER PRIMARY KEY AUTOINCREMENT'''
    for i in range(len(edgelist[0])):
        sql_str += ", " + edgelist[0][i] + " " + edgelist[1][i]
    sql_str += ");"
    cur.execute(sql_str)

    # insert data into edge table
    col_ids_str = str(edgelist[0])
    col_ids_length = len(edgelist[0])
    sql_str = "INSERT INTO EDGE " + col_ids_str + " VALUES " + _construct_sql_value_placeholder(col_ids_length)
    cur.executemany(sql_str, edgelist[2])

    # create index for table edge
    sql_str = "CREATE INDEX edge_uid on EDGE ({});".format(edgelist[0][0])
    cur.execute(sql_str)
    sql_str = "CREATE INDEX edge_oid on EDGE ({});".format(edgelist[0][1])
    cur.execute(sql_str)

    # create uid table
    sql_str = '''CREATE TABLE UID
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 uid INT NOT NULL);'''
    cur.execute(sql_str)

    # insert data into uid table
    sql_str = "INSERT INTO UID (uid) VALUES (?)"
    uid_tuple_array = [(ele,) for ele in uid_array]
    cur.executemany(sql_str, uid_tuple_array)

    # create index for column uid
    sql_str = "CREATE UNIQUE INDEX uid_uid on UID (uid);"
    cur.execute(sql_str)

    # get subgraph edges
    sql_str = "SELECT EDGE." + edgelist[0][0]
    for i in range(1, len(edgelist[0])):
        sql_str += ", EDGE.{}".format(edgelist[0][i])
    sql_str += " FROM EDGE"

    if squared == True:
        sql_str += ''', UID
                      WHERE EDGE.{} = UID.uid
                      AND EDGE.{} = UID.uid;'''.format(edgelist[0][0], edgelist[0][1])
        cur.execute(sql_str)
        subgraph = cur.fetchall()
    else:
        # create uid table
        temp_sql_str = '''CREATE TABLE OID
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           oid INT NOT NULL);'''
        cur.execute(temp_sql_str)

        # insert data into uid table
        temp_sql_str = "INSERT INTO OID (oid) VALUES (?)"
        oid_tuple_array = [(ele,) for ele in oid_array]
        cur.executemany(temp_sql_str, oid_tuple_array)

        # create index for column oid
        temp_sql_str = "CREATE UNIQUE INDEX oid_oid on OID (oid);"
        cur.execute(temp_sql_str)

        sql_str += ''', UID, OID
                      WHERE EDGE.{} = UID.uid
                      AND EDGE.{} = OID.oid;'''.format(edgelist[0][0], edgelist[0][1])
        cur.execute(sql_str)
        subgraph = cur.fetchall()

    # construct return value
    sub_edgelist = [edgelist[0], edgelist[1]]
    sub_edgelist.append(tuple(subgraph))

    # close db connection
    con.close()

    return sub_edgelist


def _construct_sql_value_placeholder(val_amount):
    if val_amount < 1:
        return None
    else:
        value_placeholder = "(?"
        value_placeholder += ",?" * (val_amount - 1)
        value_placeholder += ")"
        return value_placeholder


if __name__ == '__main__':
    tl = loadTensor('example', path='../inputData', col_types=[int, int])
