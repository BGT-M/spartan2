import functools
import logging
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as ssp
from datasketch import MinHashLSH

from . import c_MDL
from .neiminhash import NeiMinHash
from .union_find import UnionFind

from .._model import DMmodel
from spartan import STensor

logger = logging.getLogger('summarize')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

if len(logger.handlers) < 2:
    filename = os.path.join(os.path.dirname(__file__), 'summarize.log')
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


@functools.lru_cache(maxsize=10000)
def LN(n):
    return c_MDL.LN(n)


@functools.lru_cache(maxsize=10000)
def xlogx(x):
    if x == 0:
        return 0
    return x * math.log2(x)


class Summarize(DMmodel):
    B = 32
    # Maximum size of each group
    C = 500

    def __init__(self, st):
        self.sm = st.to_scipy(format='lil')
        self.sm.maximum(self.sm.T)
        self.sm.setdiag(0)
        N = self.N = self.sm.shape[0]
        self.es = self.sm.nnz // 2
        self.minhashes = [None] * N
        self.deleted = set()
        self.adj = [Counter(self.sm.rows[n]) for n in range(N)]

        self.degs = np.array(self.sm.sum(axis=1)).flatten()
        self.sizes = [1] * self.N
        self.nodes_dict = dict(zip(range(N), ({n} for n in range(N))))
        del self.sm

    def run(self, T=20):
        return self._summarize(T)

    def summarization(self, T=20):
        return self._summarize(T)

    def _merge(self, u, v):
        nodes_dict = self.nodes_dict
        sizes = self.sizes
        degs = self.degs

        nodes_dict[u] = nodes_dict[u] | nodes_dict[v]
        del nodes_dict[v]

        # Update degree and sizes
        degs[u] = degs[u] + degs[v]
        degs[v] = 0
        sizes[u] = sizes[u] + sizes[v]
        sizes[v] = 0

        uv = self.adj[u][v]
        self.adj[u].update(self.adj[v])
        self.adj[u][u] += (uv + self.adj[v][v])
        if self.adj[u][u] == 0:
            del self.adj[u][u]
        del self.adj[u][v]
        self.adj[v] = None
        for nei in self.adj[u]:
            if nei != u and nei != v and nei not in self.deleted:
                if self.adj[u][nei] != 0:
                    self.adj[nei][u] = self.adj[u][nei]
        self.deleted.add(v)

    def _update_lsh(self, threshold=0.5):
        for n in self.nodes_dict:
            if n in self.deleted:
                continue
            neighbors = [n_ for n_ in self.adj[n] if n_ not in self.deleted]
            if len(neighbors) == 0:
                continue
            m = self.minhashes[n]
            if m is None:
                m = NeiMinHash(num_perm=16, hashfunc=hash, seed=1024)
            else:
                m.clear()
            m.updates(neighbors)
            self.minhashes[n] = m

        self.lsh = MinHashLSH(threshold=threshold, num_perm=16)
        lsh = self.lsh
        with lsh.insertion_session() as session:
            for n in self.nodes_dict:
                if self.minhashes[n] is not None:
                    session.insert(str(n), self.minhashes[n])

        uf = UnionFind(self.N)
        for table in lsh.hashtables:
            d = table._dict
            for k, v in d.items():
                nodes = list(v)
                if len(nodes) <= 1:
                    continue
                n = int(nodes[0])
                for n_ in nodes[1:]:
                    uf.union(n, int(n_))
        group_dict = defaultdict(list)
        for n in self.nodes_dict:
            group_dict[uf._root(n)].append(n)
        groups = []
        for v in group_dict.values():
            while len(v) > self.C:
                groups.append(v[:self.C])
                v = v[self.C:]
            if len(v) > 0:
                groups.append(v)

        return groups

    def _merge_gain(self, u, v):
        neiu = set((n_ for n_ in self.adj[u] if n_ not in self.deleted))
        neiv = set((n_ for n_ in self.adj[v] if n_ not in self.deleted))
        du = self.degs[u]
        dv = self.degs[v]

        gain = LN(self.cnt) - LN(self.cnt-1)
        gain += 2 * (xlogx(du) + xlogx(dv) - xlogx(du+dv))
        gain += self.N * math.log2(self.cnt / (self.cnt-1))
        common_nei = 0
        for nei in neiu:
            if nei not in neiv:
                continue
            if nei == u or nei == v:
                continue
            common_nei += 1
            new_weight = self.adj[u][nei] + self.adj[v][nei]
            gain -= 2 * (xlogx(self.adj[u][nei]) + xlogx(self.adj[v][nei]))
            gain += 2 * xlogx(new_weight)
            gain += LN(self.adj[u][nei]) + LN(self.adj[v][nei])
            gain -= LN(new_weight)
        new_es = self.es - common_nei
        # Dealing with self-loop
        if u in neiu or v in neiv or u in neiv:
            new_weight = self.adj[u][u] + self.adj[v][v] + 2*self.adj[u][v]
            gain -= LN(new_weight)
            gain += xlogx(new_weight)
            if u in neiu:
                gain += LN(self.adj[u][u])
                gain -= xlogx(self.adj[u][u])
            if v in neiv:
                gain += LN(self.adj[v][v])
                gain -= xlogx(self.adj[v][v])
            if u in neiv:
                gain += LN(self.adj[u][v])
                gain -= 2 * xlogx(self.adj[u][v])
            new_es -= ((u in neiu) + (v in neiv) + (u in neiv) - 1)

        gain += LN(self.es) - LN(new_es)
        cnt = self.cnt
        gain += c_MDL.LnU(cnt*(cnt+1)//2, self.es)
        gain -= c_MDL.LnU(cnt*(cnt-1)//2, new_es)
        return gain, new_es

    def _merge_group(self, group: list):
        times = max(int(math.log2(len(group))), 1)
        merge_cnt = 0
        num_skip = 0
        gains = 0.0
        while True:
            if len(group) <= 1:
                break
            max_gain = 0
            best_pair = None
            best_new_e = 0
            for i in range(times):
                if len(group) <= 1:
                    break
                u, v = np.random.choice(group, 2)
                while u == v:
                    u, v = np.random.choice(group, 2)
                gain, new_e = self._merge_gain(u, v)
                if gain > max_gain:
                    max_gain = gain
                    best_pair = (u, v)
                    best_new_e = new_e
            if max_gain > 0 and best_pair is not None:
                u, v = best_pair
                logger.debug(f"Merge {u} and {v}, gain:{max_gain}")
                gains += max_gain
                merge_cnt += 1
                num_skip = 0
                self._merge(u, v)
                self.cnt -= 1
                self.es = best_new_e
                group.remove(v)
            else:
                num_skip += 1

            if num_skip >= times:
                break
        return merge_cnt, gains

    def _summarize(self, T):
        N = self.N

        degs = self.degs
        degs_orig = degs.copy()
        nodes_dict = self.nodes_dict

        M_threshold = 0.75
        m_threshold = 0.35
        x = (m_threshold / M_threshold) ** (1/(T-1))
        start_time = time.time()
        t = 0
        self.N = self.cnt = N

        self.total_gain = 0.0
        while t < T:
            t += 1
            threshold = M_threshold * (x**(t-1))
            logger.info(f"Iteration {t}, threshold: {threshold}")
            start_time2 = time.time()
            groups = self._update_lsh(threshold)
            elapsed = time.time() - start_time2
            merge_cnt = 0
            for group in groups:
                tmp1, tmp2 = self._merge_group(group)
                merge_cnt += tmp1
                self.total_gain += tmp2
            logger.info(f"Merge {merge_cnt} in Iteration {t}")
            if merge_cnt == 0:
                logger.info(f"No improvment at iteration {t}, break.")
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Summarize {N} nodes to {self.cnt} nodes with {self.es} edges, \
                 costs {elapsed} seconds")
        logger.info(f"Total gain: {self.total_gain}")

        # Constructing P and P_
        rows, cols, datas = [], [], []
        for i, (n, nodes) in enumerate(nodes_dict.items()):
            assert len(nodes) != 0
            D = sum(degs_orig[n_] for n_ in nodes)
            nodes = list(nodes)
            rows.extend([i] * len(nodes))
            cols.extend(nodes)
            if D == 0:
                datas.extend([1.0 / len(nodes)] * len(nodes))
            else:
                datas.extend((degs_orig[n_] / D for n_ in nodes))
        n = len(nodes_dict)

        supernodeid = dict((n, i) for i, n in enumerate(nodes_dict))
        rows, cols, datas = [], [], []
        for i, n_ in enumerate(nodes_dict):
            for nei in self.adj[n_]:
                if nei not in supernodeid:
                    continue
                rows.append(i)
                cols.append(supernodeid[nei])
                datas.append(self.adj[n_][nei])
        sm_s = ssp.coo_matrix((datas, (rows, cols)), shape=(n, n))

        self.nodes_dict = nodes_dict
        self.sm_s = sm_s
        return nodes_dict, STensor.from_scipy_sparse(sm_s)
