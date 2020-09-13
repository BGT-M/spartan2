#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Desc    :   Definition of graph structure.
'''

# here put the import lib
from . import STensor
import spartan as st
import numpy as np


class Graph:
    def __init__(self, graph_tensor: STensor, weighted: bool = False,
                 bipartite: bool = False, modet=None):
        '''Construct a graph from sparse tensor.
        If the sparse tensor has more than 2 modes, then it is a rich graph.
        Parameters:
        ------
        modet: int
            The order of mode in graph tensor for temporal bins if exit, start from zero.
            Default is 3.
        '''
        self.graph_tensor = graph_tensor
        self.weighted = weighted
        self.bipartite = bipartite
        self.modet = modet  # which mode is time dimension
        self.nprop = graph_tensor.ndim - 2  # num of edge properties

        self.sm = graph_tensor.sum_to_scipy_sparse(modes=(0, 1))
        if not weighted:
            self.sm = (self.sm > 0).astype(int)
        if not bipartite:
            self.sm = self.sm.maximum(self.sm.T)

    def get_time_tensor(self):
        '''Get the tensor only have time dimension.
        If nprop == 1 and modet == 3, then the tensor is graph_tensor itself.
        If modet is None, then None is returned.
        '''
        return self.get_one_prop_tensor(self.modet)

    def get_one_prop_tensor(self, mode):
        '''Get the tensor only have one edge-property dimension.
        if nprop == 1 and mode == 3, then the tensor is graph_tensor itself.
        If mode is None, and other invalidation, then None is returned.
        '''
        if self.nprop == 1 and mode == 3:
            return graph_tensor
        elif self.nprop > 1 and mode is not None and\
                mode < self.nprop + 2:
            return STensor((self.graph_tensor.coords[(0, 1, mode), :],
                            self.graph_tensor.data))
        else:
            return None

    def get_sub_graph(self, rows, cols):

        cootensor = self.graph_tensor

        gr = -1 * st.ones(cootensor.shape[0], dtype=int)
        gc = -1 * st.ones(cootensor.shape[1], dtype=int)

        lr = len(rows)
        lc = len(cols)

        ar = st.arange(0, lr, 1)
        ac = st.arange(0, lc, 1)
        gr[rows[ar]] = ar
        gc[cols[ac]] = ac
        mrow = cootensor.coords[0]
        mcol = cootensor.coords[1]
        newelem = (gr[mrow] > -1) & (gc[mcol] > -1)

        newrows = mrow[newelem]
        newcols = mcol[newelem]

        subcoords = st.stack((gr[newrows], gc[newcols],
            *cootensor.coords[2:,newelem]), axis=0)
        subvalues = cootensor.data[newelem]

        subtensor = st.STensor((subcoords, subvalues),
                shape=(lr,lc,*cootensor.shape[2:]) )

        return st.Graph(subtensor, self.weighted, self.bipartite, self.modet)

    def get_subgraph_nedges(self, rows, cols):
        """
        Pulls out an arbitrary i.e. non-contiguous submatrix out of
        a sparse.coo_matrix.

        Returns
        ------
        tuples of org_row_id, org_col_id, value
        """
        matr = self.sm.tocoo()

        gr = -1 * st.ones(matr.shape[0], dtype=int)
        gc = -1 * st.ones(matr.shape[1], dtype=int)

        lr = len(rows)
        lc = len(cols)

        ar = st.arange(0, lr, 1)
        ac = st.arange(0, lc, 1)
        gr[rows[ar]] = ar
        gc[cols[ac]] = ac
        mrow = matr.row
        mcol = matr.col
        newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
        subvalues = matr.data[newelem]

        if self.weighted:
            nedges = len(subvalues)
        else:
            nedges = subvalues.sum()

        return nedges

    def degrees(self):
        rowdegs, coldegs = self.sm.sum(axis=1), self.sm.sum(axis=0)
        return rowdegs.A1, coldegs.A1

    def get_edgelist_array(self):
        """
        Get the edgelist of graph (without edge attributes),
        summing up weights of the same edge.

        Returns
        ------
        2D numpy ndarray of (row, col, weights) as list.
        """
        coosm = self.sm.tocoo()
        data = coosm.data
        row = coosm.row
        col = coosm.col
        return np.vstack((row, col, data)).T


