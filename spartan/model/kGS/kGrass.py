import math
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import scipy.sparse as ssp

import spartan as st
from .._model import DMmodel
from spartan import STensor


class kGS(DMmodel):
    def __init__(self, graph) -> None:
        self.sm = graph.sm.tolil()
        self.sm.setdiag(0)
        self.N = self.sm.shape[0]
        self.nodes = list(range(self.N))

        self.adj = [Counter(self.sm.rows[n]) for n in range(self.N)]
        self.sizes = [1] * self.N
        self.nodes_dict = dict(
            zip(range(self.N), ({n} for n in range(self.N))))

    def _greedy(self):
        min_pair, min_loss = None, math.inf
        for u, v in combinations(self.nodes):
            loss = self._merge_loss(u, v)
            if loss < min_loss:
                min_loss = loss
                min_pair = (u, v)
        return min_pair, min_loss

    def _sample_pairs(self, C=0):
        if callable(C):
            C = C()
        elif C == 0:
            C = int(math.log2(self.numNode))
        C = int(C)

        min_pair, min_loss = None, math.inf
        for _ in range(C):
            u, v = np.random.choice(self.nodes, 2)
            loss = self._merge_loss(u, v)
            if loss < min_loss:
                min_loss = loss
                min_pair = (u, v)
        return min_pair, min_loss

    def _linear_check(self):
        u = np.random.choice(self.nodes, 1)[0]

        min_pair, min_loss = None, math.inf
        for v in self.nodes:
            if v == u:
                continue
            loss = self._merge_loss()
            if loss < min_loss:
                min_loss = loss
                min_pair = (u, v)
        return min_pair, min_loss

    def run(self, K, strategy='sample_pairs'):
        return self._summarize(K, strategy)

    def summarization(self, K, strategy='sample_pairs'):
        return self._summarize(K, strategy)

    def _check_parameters(self, K, strategy):
        if K >= self.N:
            print(f"`K`({K}) should be less than size of graph({self.N})")
            return False
        if strategy not in ('greedy', 'linear_check', 'sample_pairs'):
            print(f"`Strategy`({strategy}) must be of: 'greedy', 'linear_check', 'sample_pairs'")
            return False
        if strategy == 'greedy' and self.N >= 10000:
            warnings.warn("Using greedy strategy on large graphs is time-consuming, try using other strategies instead.")
        return True


    def _summarize(self, K, strategy='sample_pairs'):
        if not self._check_parameters(K, strategy):
            print("Check parameter fails")
            return
        self.numNode = self.N
        while self.numNode > K:
            pair = None
            if strategy == 'greedy':
                pair, loss = self._greedy()
            elif strategy == 'linear_check':
                pair, loss = self._linear_check()
            elif strategy == 'sample_pairs':
                pair, loss = self._sample_pairs()
            if pair is None:
                break
            u, v = pair
            self._merge(u, v)

        l1_error = self._final_errors()
        print(f"Summarize from {self.N} nodes to {self.numNode} nodes.")
        print(f"Average L1 loss: {l1_error / (self.N * self.N):.4f}")

        rows, cols, datas = [], [], []
        for i, (n, nodes) in enumerate(self.nodes_dict.items()):
            datas.extend([1.0 / len(nodes)] * len(nodes))

        rows, cols, datas = [], [], []
        for i, (n, nodes) in enumerate(self.nodes_dict.items()):
            assert len(nodes) != 0
            nodes = list(nodes)
            rows.extend([i] * len(nodes))
            cols.extend(nodes)
            datas.extend([1.0 / len(nodes)] * len(nodes))

        P_ = ssp.csr_matrix(([1] * len(datas), (rows, cols)), shape=(len(self.nodes_dict), self.N))
        sm_s = P_ @ self.sm @ P_.T

        self.nodes_dict = self.nodes_dict
        self.sm_s = sm_s

        return STensor.from_scipy_sparse(sm_s)

    def _merge_loss(self, u, v):
        sizeu = self.sizes[u]
        sizev = self.sizes[v]
        if sizeu == 0 or sizev == 0:
            return math.inf

        loss = 0.0
        new_e = 0
        if u in self.adj[u]:
            eu = self.adj[u][u]
            new_e += eu
            loss -= 8.0 * eu * eu / (sizeu * (sizeu-1))
        if v in self.adj[v]:
            ev = self.adj[v][v]
            new_e += ev
            loss -= 8.0 * ev * ev / (sizev * (sizev-1))

        for n in self.adj[u]:
            value = self.adj[u][n]
            if n == u or value == 0 or self.sizes[n] == 0:
                continue
            loss -= 4.0 * value * value / (self.sizes[u] * self.sizes[n])
        for n in self.adj[v]:
            value = self.adj[v][n]
            if n == u or value == 0 or self.sizes[n] == 0:
                continue
            loss -= 4.0 * value * value / (self.sizes[v] * self.sizes[n])

        if v in self.adj[u]:
            new_e += 2 * self.adj[u][v]
        loss += 8 * new_e * new_e / ((sizeu+sizev) * (sizeu + sizev-1))

        common_neis = self.adj[u].keys() & self.adj[v].keys()
        for n in common_neis:
            if n == u or n == v or self.sizes[n] == 0:
                continue
            eui = self.adj[u][n]
            evi = self.adj[v][n]
            if eui == 0 and evi == 0:
                continue
            loss += 4.0 / (sizeu + sizev) * (eui * evi + evi *
                                             evi + 2.0 * eui * evi) / self.sizes[n]

        return loss

    def _merge(self, u, v):
        nodes_dict = self.nodes_dict

        nodes_dict[u] = nodes_dict[u] | nodes_dict[v]
        del nodes_dict[v]
        self.sizes[u] += self.sizes[v]
        self.sizes[v] = 0

        # Update adj list
        uv = self.adj[u][v]
        self.adj[u].update(self.adj[v])
        self.adj[u][u] += (uv + self.adj[v][v])
        if self.adj[u][u] == 0:
            del self.adj[u][u]
        del self.adj[u][v]
        self.adj[v] = None
        for nei in self.adj[u]:
            if nei != u and nei != v and self.sizes[nei] != 0:
                if self.adj[u][nei] != 0:
                    self.adj[nei][u] = self.adj[u][nei]

        self.nodes.remove(v)
        self.numNode -= 1

    def _final_errors(self):
        l1_error = 0.0
        for n in self.nodes:
            sizesn = self.sizes[n]
            if sizesn == 0:
                continue
            for nei in self.adj[n]:
                sizes_nei = self.sizes[nei]
                if sizes_nei == 0:
                    continue
                if nei == n:
                    w = self.adj[n][n] / (sizesn * (sizesn - 1))
                    l1_error += (1-w)
                else:
                    w = self.adj[n][nei] / (sizesn * sizes_nei)
                    l1_error += (1-w)

        l1_error *= 2
        for n in self.nodes:
            for nei in self.adj[n]:
                if self.sizes[nei] == 0:
                    continue
                l1_error += self.adj[n][nei]

        return l1_error