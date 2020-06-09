import functools
import logging
import math
import os
import pickle
import shutil
import time
from collections import defaultdict

import numpy as np
import scipy as sp
import scipy.io
import scipy.sparse as ssp
from datasketch import MinHashLSH, MinHash

from . import c_MDL
from PriorityTree import PriorityTree, PTNode

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


class Summarizer:
    B = 8
    C = 10

    def __init__(self, sm):
        self.sm = sm.tolil()

    def summarize(self, dataset, output_dir):
        lilm = self.sm
        N = lilm.shape[0]
        M = lilm.sum() // 2
        nnz = M

        degs = np.array(lilm.sum(axis=1))
        degs = np.squeeze(degs).tolist()
        pt = PriorityTree(((d, i) for i, d in enumerate(degs)))
        sizes = defaultdict(lambda: 1)
        nodes = {}
        for n in range(N):
            nodes[n] = {n}

        re_error = 0.0
        length = LN(N)
        length += sum(LN(d) for d in degs)
        length += N * LN(1)
        length += c_MDL.LnU(N*(N+1) // 2, M)
        length += M * LN(1)
        logger.info(f"Init length: {length}")

        # Initialize MinHash LSH
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        lsh_path = os.path.join(output_dir, f'{dataset}.lsh')
        if os.path.exists(lsh_path):
            lsh, minhashes = pickle.load(open(lsh_path, 'rb'))
        else:
            lsh = MinHashLSH(threshold=0.4, num_perm=64)
            minhashes = [None] * N
            for n in range(N):
                m = MinHash(num_perm=64, hashfunc=hash, seed=1024)
                neighbors = ssp.find(lilm[n])[1]
                for nei in neighbors:
                    m.update(nei)
                minhashes[n] = m
                lsh.insert(str(n), m)
            pickle.dump([lsh, minhashes], open(lsh_path, 'wb'))

        start_time = time.time()
        cnt = N
        end = False
        non_gain = 0

        while not end:
            logger.debug(f"Iteration {N-cnt}")

            while True:
                du, u = pt.pop()
                if du == math.inf:
                    logger.debug(f"Choose node {u} and break")
                    end = True
                    break
                if du != degs[u]:
                    logger.warning(f"Degree not match! {du} != {degs[u]}")
                candidates = lsh.query(minhashes[u])
                candidates = candidates[:min(self.C, len(candidates))]

                if len(candidates) == 1:
                    non_gain += 1
                    if non_gain >= cnt:
                        end = True
                        break
                    logger.debug(f"No candidate for node {u}({du})")
                    continue
                if not end:
                    logger.debug(f"Choose node: {u} with degree {du}, {len(candidates)} candidates.")
                break
            if end:
                break

            neiu = set(ssp.find(lilm[u])[1])
            max_, v = 0, -1

            len_matrix = c_MDL.LnU(cnt*(cnt-1) // 2, nnz)
            g_common_nei = 0
            g_cost = 0
            for c in candidates:
                c = int(c)
                if c == u:
                    continue
                dc = degs[c]

                gain = du * math.log2(du) + dc * math.log2(dc)
                gain -= (du + dc) * math.log2(du + dc)
                gain *= 2

                gain2 = 0
                common_nei = 0
                for nei in neiu:
                    if lilm[c, nei] == 0:
                        continue
                    common_nei += 1
                    gain -= lilm[u, nei] * math.log2(lilm[u, nei])
                    gain -= lilm[c, nei] * math.log2(lilm[c, nei])
                    new_weight = lilm[u, nei] + lilm[c, nei]
                    gain += new_weight * math.log2(new_weight)

                    gain2 += LN(lilm[u, nei]) + \
                        LN(lilm[c, nei]) - LN(new_weight)

                cost = -gain
                gain += gain2
                gain += LN(cnt) - LN(cnt-1)
                size_u, size_v = sizes[u], sizes[v]
                gain += LN(size_u)+LN(size_v)-LN(size_u+size_v)
                gain += c_MDL.log_comb(size_u+size_v, size_u)
                new_nnz = nnz - common_nei
                # gain += common_nei * self.B
                gain -= c_MDL.LnU((cnt-1)*(cnt-2) // 2, new_nnz)
                gain += len_matrix

                if gain > max_:
                    max_ = gain
                    g_common_nei = common_nei
                    v = c
                    g_cost = cost
            if end:
                break
            if v == -1:
                logger.debug(f"No non-negative gain for node {u}")
                non_gain += 1
                if non_gain >= cnt:
                    end = True
                continue
            non_gain = 0
            re_error += g_cost

            logger.debug(f"Merge {u} and {v}, gain: {max_}, cost:{g_cost}")
            nodes[u] = nodes[u] | nodes[v]
            if v in nodes:
                del nodes[v]
            length -= gain
            logger.debug(f"Current length: {length}")
            neiv = set(ssp.find(lilm[v])[1])

            # Update degree and sizes
            pt.update(u, (degs[u] + degs[v], u))
            degs[u] = degs[u] + degs[v]
            pt.update(v, (math.inf, v))
            degs[v] = 0
            sizes[u] = sizes[u] + sizes[v]
            sizes[v] = 0

            lilm[u] = lilm[u] + lilm[v]
            lilm[u, u] += lilm[u, v]
            for nei in ssp.find(lilm[u])[1]:
                if nei != u and nei != v:
                    lilm[nei, u] = lilm[u, nei]
            lilm[v] = 0
            lilm[:, v] = 0
            cnt -= 1
            nnz -= g_common_nei

            # Update LSH
            mu = minhashes[u]
            for nei in (neiu | neiv):
                if nei not in neiu:
                    mu.update(nei)
                if nei == u or nei == v:
                    continue
                # Leave v in nei's MinHash
                if nei not in neiu:
                    m = minhashes[nei]
                    m.update(u)
                    if str(nei) in lsh:
                        lsh.remove(str(nei))
                    lsh.insert(str(nei), m)
                    minhashes[nei] = m

            if str(u) in lsh:
                lsh.remove(str(u))
            if str(v) in lsh:
                lsh.remove(str(v))
            lsh.insert(str(u), mu)
            minhashes[u] = mu
            minhashes[v] = None

        elapsed = time.time() - start_time
        logger.info(f"Summarize {N} nodes to {cnt} nodes, costs {elapsed} seconds, final length: {length}")
        logger.info(f"Total cost: {re_error}/{re_error/N}")

        output_file = './output/summaried.mat'.format(dataset)
        scipy.io.savemat(output_file, {'sm': lilm})
        output_file = './output/{}/nodes.dict'.format(dataset)
        pickle.dump(nodes, open(output_file, 'wb'))

        return lilm, nodes