#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Desc    :   Definition of graph structure.
'''

# here put the import lib
from . import STensor


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

    def degrees(self):
        rowdegs, coldegs = self.sm.sum(axis=1), self.sm.sum(axis=0)
        return rowdegs, coldegs.T
