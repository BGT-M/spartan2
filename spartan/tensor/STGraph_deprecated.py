import os
class STGraph:
    def __init__(self, sm, weighted, bipartite, rich=False, attlist=None, relabel=False, labelmaps=(None, None)):
        '''
            sm: sparse adj matrix of (weighted) graph
            weighted: graph is weighte or not
            attlist: attribute list with edges, no values
            relabel: relabel or not
            labelmaps: label maps from old to new, and inverse maps from new to
            old
        '''
        self.sm = sm
        self.weighted = weighted
        self.rich = rich
        self.attlist = attlist
        self.relabel = relabel
        self.labelmaps = labelmaps
        self.bipartite = bipartite

    def degrees(self):
        rowdegs, coldegs = self.sm.sum(axis=1), self.sm.sum(axis=0)
        return rowdegs, coldegs.T
