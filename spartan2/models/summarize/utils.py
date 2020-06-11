import os

import numpy as np
import scipy as sp
import scipy.sparse as ssp
from collections import defaultdict


def read_edgelist(path, comments='%', delimeter=' ', undir=True, sm_type='csr'):
    node2idx = dict()
    rows, cols, values = [], [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(comments) or len(line) == 0:
                continue

            elems = line.split(delimeter)
            source, target = elems[0], elems[1]
            try:
                source, target = int(source), int(target)
            except ValueError:
                pass

            if source not in node2idx:
                node2idx[source] = len(node2idx)
            if target not in node2idx:
                node2idx[target] = len(node2idx)

            rows.append(node2idx[source])
            cols.append(node2idx[target])
            if len(elems) > 2:
                values.append(elems[2])
            else:
                values.append(1)

    N = len(node2idx)
    coom = ssp.coo_matrix((values, (rows, cols)), shape=(N, N), dtype=np.int32)
    if undir:
        coomt = coom.T
        coom = coom.maximum(coomt)
        for i, j, v in zip(*ssp.find(coom)):
            if i == j:
                coom[i, j] = 2 * v
    if sm_type == 'csr':
        sm = coom.tocsr()
    elif sm_type == 'csc':
        sm = coom.tocsc()
    elif sm_type == 'coo':
        sm = coom
    elif sm_type == 'lil':
        sm = coom.tolil()
    else:
        print("Only support csr/csc/coo/lil currently")
        sm = coom

    if sm_type == 'csr' or sm_type == 'csc':
        sm.sort_indices()
    return sm, node2idx
