#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import copy


# third-part libs
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg

# project
from .priority_queue import PriorQueMin


def list2sm_bip(srcs, dests):
    m = max(srcs) + 1
    n = max(dests) + 1
    M = sps.csc_matrix(([1]*len(srcs), (srcs, dests)), shape=(m, n))
    sm = M > 0
    return sm.astype('int')


def list2sm_mono(srcs, dests, issym=True, n=None):
    if n is None:
    	n = max([max(srcs), max(dests)]) + 1
    M = sps.csc_matrix(([1]*len(srcs), (srcs, dests)), shape=(n, n))
    sm = M > 0
    if issym:
        sm += sm.transpose()
    return sm.astype('int')


def c2score(mat, rowset, colset):
    return mat[list(rowset), :][:, list(colset)].sum(axis=None)


def avgdeg_even(mat, alpha=None):
    (m, n) = mat.shape
    return fast_greedy_decreasing(mat, [1.0] * n)

def avgdeg_log(mat, alpha=5):
    (m, n) = mat.shape
    coldeg = np.squeeze(mat.sum(axis=0).A)
    colwt = 1.0 / np.log(coldeg + alpha)
    col_diag = sps.lil_matrix((n, n))
    col_diag.setdiag(colwt)
    return fast_greedy_decreasing(mat * col_diag, [1.0] * n) # colwt

def avgdeg_sqrt(mat, alpha=5):
    (m, n) = mat.shape
    coldeg = np.squeeze(mat.sum(axis=0).A)
    colwt = 1.0 / np.sqrt(coldeg + alpha)
    col_diag = sps.lil_matrix((n, n))
    col_diag.setdiag(colwt)
    return fast_greedy_decreasing(mat * col_diag, [1.0] * n)  # colwt

def fast_greedy_decreasing(mat, colweights=None):
    # return the subgraph with optimal (weighted) degree density using Charikai's greedy algorithm
    (m, n) = mat.shape
    if colweights is None:
        colweights = [1.0] * n

    ml = mat.tolil()
    mlt = mat.transpose().tolil()
    row_set = set(range(0, m))
    col_set = set(range(0, n))

    final_rows,final_cols = copy.copy(row_set), copy.copy(col_set)
    # print(len(row_set), len(col_set))

    cur_score = c2score(mat, row_set, col_set)
    best_avgscore = cur_score * 1.0 / (len(row_set) + len(col_set))
    # best_sets = (row_set, col_set)
    #print("finished setting up greedy, init score: {}".format(best_avgscore))

    # *decrease* in total weight when *removing* this row / column
    row_deltas = np.squeeze(1.0*mat.sum(axis=1).A)
    col_deltas = np.squeeze(1.0*mat.sum(axis=0).A)
    #print("finished setting deltas")

    row_tree = PriorQueMin(row_deltas)
    col_tree = PriorQueMin(col_deltas)
    #print("finished building min trees")

    n_dels = 0
    deleted = list()
    best_n_dels = 0

    while row_set and col_set:
        if (len(col_set) + len(row_set)) % 5000000 == 0:
            print("   PROC: current set size = {}".format(len(col_set) + len(row_set)))
        (del_row, row_delt) = row_tree.getMin()
        (del_col, col_delt) = col_tree.getMin()
        if row_delt <= col_delt:  # remove this row
            cur_score -= row_delt
            for j in ml.rows[del_row]:
                delt = colweights[j] * ml[del_row, j]
                col_tree.change(j, -1 * delt)
            row_set -= {del_row}
            row_tree.change(del_row, float('inf'))
            deleted.append((0, del_row))
        else:                     # remove this column
            cur_score -= col_delt
            for i in mlt.rows[del_col]:
                delt = colweights[del_col] * ml[i, del_col] #mlt[del_col, i]
                row_tree.change(i, -1 * delt)
            col_set -= {del_col}
            col_tree.change(del_col, float('inf'))
            deleted.append((1, del_col))

        n_dels += 1
        cur_avgscore = cur_score * 1.0 / (len(col_set) + len(row_set))

        if cur_avgscore > best_avgscore:
            best_avgscore = cur_avgscore
            best_n_dels = n_dels
            # best_sets = (row_set, col_set)

    # reconstruct the best row and column sets
    for i in range(best_n_dels):
        r_or_c, nd_id = deleted[i]
        if r_or_c == 0:
            final_rows.remove(nd_id)
        else:
            final_cols.remove(nd_id)
    return (list(final_rows), list(final_cols)), best_avgscore

def fast_greedy_decreasing_monosym(mat):
    # return the subgraph with optimal (weighted) degree density using Charikai's greedy algorithm
    (m, n) = mat.shape
    #uprint((m, n))
    assert m == n
    ml = mat.tolil()
    node_set = set(range(0, m))
    # print(len(node_set))
    final_ = copy.copy(node_set)
    
    cur_score = c2score(mat, node_set, node_set)
    best_avgscore = cur_score * 1.0 / len(node_set)
    # best_sets = node_set
    #print("finished setting up greedy, init score: {}".format(best_avgscore / 2.0))

    # *decrease* in total weight when *removing* this row / column
    delta = np.squeeze(1.0*mat.sum(axis=1).A)
    tree = PriorQueMin(delta)
    #print("finished building min trees")

    n_dels = 0
    deleted = list()
    best_n_dels = 0

    while len(node_set) > 1:
        if len(node_set) % 500000 == 0:
            print("   PROC: current set size = {}".format(len(node_set)))
        (delidx_, delt_) = tree.getMin()
        cur_score -= delt_ * 2
        for j in ml.rows[delidx_]:   # remove this row / column
            tree.change(j, -1.0 * ml[delidx_, j])

        tree.change(delidx_, float('inf'))
        node_set -= {delidx_}
        deleted.append(delidx_)

        n_dels += 1
        if n_dels < n:
            cur_avgscore = cur_score * 1.0 / len(node_set)
            if cur_avgscore > best_avgscore:
                best_avgscore = cur_avgscore
                best_n_dels = n_dels

    # reconstruct the best row and column sets
    for i in range(best_n_dels):
        nd_id = deleted[i]
        final_.remove(nd_id)

    return list(final_), best_avgscore


def spectral_levels(sm, topk=1, scale=1.0):
    (m, n) = sm.shape
    U, S, V = linalg.svds(sm.asfptype(), k=topk, which='LM')
    U, V = U[:, ::-1], V.T[:, ::-1]
    rows, cols = list(), list()
    
    for i in range(topk):
        if abs(max(U[:, i])) < abs(min(U[:, i])):
            U[:, i] *= -1
        rows.append(list(np.where(U[:, i] >= 1.0 * scale / np.sqrt(m))[0]))
        
        if abs(max(V[:, i])) < abs(min(V[:, i])):
            V[:, i] *= -1
        cols.append(list(np.where(V[:, i] >= 1.0 * scale / np.sqrt(n))[0]))
    
    if topk > 1:
        S = S[::-1]
        #print("top {} singular values: {}".format(topk, S))
    
    return rows, cols, S


def spectral_levels_asym(sm, topk=1, scale=1.0, s=2, p=5):
    (m, n) = sm.shape
    A = sps.bmat([[None, sm],[sm.T, None]], 'csc')
    S, U = freigs(A.astype(float), topk, s=topk, p=p, which='LA')
    #S, U = freigs(A.astype(float), topk, s, p, which='LM')
    sorted_idx = np.argsort(np.abs(S))[::-1]
    U = U[:, sorted_idx]
    rows, cols = list(), list()

    for i in range(topk):
        if abs(max(U[:m, i])) < abs(min(U[:m, i])):
            U[:m, i] *= -1
        rows.append(list(np.where(U[:m, i] >= 1.0 * scale / np.sqrt(m))[0]))

        if abs(max(U[m:, i])) < abs(min(U[m:, i])):
            U[m:, i] *= -1
        cols.append(list(np.where(U[m:, i] >= 1.0 * scale / np.sqrt(n))[0]))

    return rows, cols, np.abs(S)[sorted_idx]
