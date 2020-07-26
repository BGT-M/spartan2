#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Desc    :   Definition of graph structure.
'''

# here put the import lib
from . import STensor


class Graph:
    def __init__(self, graph_tensor: STensor, weighted: bool = False, bipartite: bool = False):
        '''Construct a graph from sparse tensor.
        If the sparse tensor has more than 2 modes, then it is a rich graph.
        '''
        self.graph_tensor = graph_tensor
        self.weighted = weighted
        self.bipartite = bipartite

        self.sm = graph_tensor.sum_to_scipy_sparse(modes=(0, 1))
        if not weighted:
            self.sm = (self.sm > 0).astype(int)
        if not bipartite:
            self.sm = self.sm.maximum(self.sm.T)

    def degrees(self):
        rowdegs, coldegs = self.sm.sum(axis=1), self.sm.sum(axis=0)
        return rowdegs, coldegs.T
