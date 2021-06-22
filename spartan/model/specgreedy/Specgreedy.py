# sys
import os
import sys
import time
import argparse
import gc

# third-part libs
import numpy as np
import scipy.sparse.linalg as linalg
import scipy.sparse

# project
from spartan.model.fraudar.greedy import logWeightedAveDegree, sqrtWeightedAveDegree, aveDegree, fast_greedy_decreasing_monosym
from .._model import DMmodel


class Specgreedy(DMmodel):
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, out_path = "./", file_name = "specgreedy", bipartite=False, T=5, delete_type="edge", **kwargs):
        Mcur = self.data.to_scipy().tocsr()
        res = []
        t = 0
        print("Specgreedy for find %d dense subgraphs." % T)
        while (t < T):
            print("\nRunning for %d-th subgraph" % t)
            if bipartite:
                list_row, list_col, score = self.run_bip(Mcur, **kwargs)
            else:
                list_row, list_col, score = self.run_undi(Mcur, **kwargs)
            res.append((list_row, list_col, score))

            np.savetxt("%s_%s.rows" % (out_path + file_name, t), np.array(list_row).reshape(-1, 1), fmt='%d')
            np.savetxt("%s_%s.cols" % (out_path + file_name, t), np.array(list_col).reshape(-1, 1), fmt='%d')
            print("score obtained is ", score)

            if (len(list_row) == 0 or len(list_col) == 0):
                print("Finding subgraph failed. Specgreedy stopped.")
                break

            t += 1
            gc.collect()

            if (t >= T):
                break
            
            tic = time.time()
            if delete_type == "edge":
                ## only delete inner connections
                (rs, cs) = Mcur.nonzero() # (u, v)
                rowSet = set(list_row)
                colSet = set(list_col)
                for i in range(len(rs)):
                    if rs[i] in rowSet and cs[i] in colSet:
                        Mcur[rs[i], cs[i]] = 0
            elif delete_type == "node":
                diag = scipy.sparse.eye(Mcur.shape[0]).tolil()
                for r in list_row:
                    diag[r, r] = 0
                diag = diag.tocsr()
                Mcur = diag.dot(Mcur)

                diag = scipy.sparse.eye(Mcur.shape[1]).tolil()
                for c in list_col:
                    diag[c, c] = 0
                diag = diag.tocsr()
                Mcur = Mcur.dot(diag)
            else:
                raise ValueError("Invalid argument delete_type. Please set 'edge' or 'node'")
            toc = time.time()
            print("Remove current block: %ss" % (toc-tic))

        return res
            

    def run_undi(self, sm, weighted = True, topk = 5, scale = 1.0):
        # outfn = output_path
        w_g = weighted
        
        # print("## Dataset: {}".format(infn[infn.rfind('/')+1:]))
        greedy_fun = fast_greedy_decreasing_monosym

        t0 = time.time()
        ms, ns = sm.shape
        n = max(ms, ns)
        sm -= sps.diags(sm.diagonal())
        es = sm.sum()

        if (abs(sm-sm.T)>1e-10).nnz > 0:
            sm += sm.T
        if not w_g:
            print("max edge weight: {}".format(sm.max()))
            sm = sm > 0
            sm = sm.astype('int')

        print("load graph @ {}s".format(time.time() - t0))
        print("graph: #node: {}, #edge: {}, # es: {}".format((ms, ns), es, sm.sum()))
        print("matrix max: {}, min: {}, shape: {}\n".format(sm.max(), sm.min(), (n ,n)))
        print("Finding subgraph with top k singular values:", topk)

        orgnds, cans = None, None
        opt_density = 0.0
        opt_k = -1

        k = 0
        decom_n = 0

        start = 3
        step = 3
        isbreak = False
        t1 = time.time()
        while k < topk:
            print("\nComputing top{} singular vectors and values for efficency".format(start + decom_n * step))
            U, S, V = linalg.svds(sm.asfptype(), k=start + decom_n * step, which='LM', tol=1e-4)
            U, S, V = U[:, ::-1], S[::-1], V.T[:, ::-1]
            print("lambdas:", S)
            kth  = k
            while kth < start + decom_n * step - 1 and kth < topk:
                if abs(max(U[:, kth])) < abs(min(U[:, kth])): U[:, kth] *= -1
                if abs(max(V[:, kth])) < abs(min(V[:, kth])): V[:, kth] *= -1
                row_cans = list(np.where(U[:, kth] >= 1.0 / np.sqrt(ms))[0])
                # col_cans = list(np.where(V[:, kth] >= 1.0 / np.sqrt(ns))[0])
                col_cans = row_cans
                if len(row_cans) <= 1 or len(col_cans) <= 1:
                    print("SKIP: candidates sizes are too small: {}".format((len(row_cans), len(col_cans))))
                    kth += 1
                    k += 1
                    continue
                sm_part = sm[row_cans, :][:, col_cans]
                # print("{}, size: {}".format(kth, sm_part.shape))
                row_ids, col_ids, avgsc_part = greedy_fun(sm_part)
                kth += 1
                k += 1
                gc.collect()

                print("k_cur:{}, size: {}, density: {}.  @ {}s\n".format(kth, (len(row_ids), len(col_ids)),  avgsc_part, time.time() - t1))
                if avgsc_part > opt_density:
                    opt_k, opt_density = kth, avgsc_part
                    sm_pms = max(len(row_cans), len(col_cans))
                    cans = row_cans
                    fin_pms = len(nds_res)
                    print("Update. svd init shape (candidates size): {}".format((sm_pms, sm_pms)))
                    print("Update. size: {}, score: {}\n".format((fin_pms, fin_pms), avgsc_part))
                    nd_idx = dict(zip(range(sm_pms), sorted(cans)))
                    orgnds = [nd_idx[id] for id in nds_res]

                if 2.0*opt_density >= S[kth]: # kth < topk and
                    print("Early Stopped. k_cur: {},  optimal density: {}, lambda_k: {}".format(kth, opt_density, S[kth]))
                    isbreak = True
                    break
            if isbreak:
                break
            decom_n += 1


        if orgnds is not None:
            print("\noptimals: k:{}, size:{}, density:{}".format(opt_k, fin_pms, opt_density))
            print("total time @ {}s".format(time.time() - t1))
            return orgnds, orgnds, opt_density
        else:
            print("No dense subgraphs found.")
            return [], [], 0

    def run_bip(self, sm, weighted = True, topk = 5, alpha = 5.0, scale = 1.0, col_wt = "even", maxsize = -1):
        # outfn = output_path
        w_g = weighted

        #alpha = 1.0
        greedy_func = None
        if col_wt == 'even':
            greedy_func = logWeightedAveDegree
        elif args.col_wt == 'sqrt':
            greedy_func = sqrtWeightedAveDegree
        else:
            greedy_func = aveDegree

        t0 = time.time()
        ms, ns = sm.shape
        if not w_g:
            print("max edge weight: {}".format(sm.max()))
            sm = sm > 0
            sm = sm.astype('int')
        es = sm.sum()
        print("load graph time @ {}s".format(time.time() - t0))
        print("graph: #node: {},  #edge: {}".format((ms, ns), es))
        print("matrix max: {}, min: {}, shape: {}\n".format(sm.max(), sm.min(), sm.shape))
        print("Finding subgraph with top k singular values:", topk)

        opt_k = -1
        opt_density = 0.0
        orgnds, cans = None, None
        fin_pms, fin_pns = 0, 0

        k = 0
        decom_n = 0

        start = 3
        step = 3
        isbreak = False
        orgnds = None

        t1 = time.time()  
        while k < topk:
            print("\nComputing top{} singular vectors and values for efficency".format(start + decom_n * step))
            U, S, V = linalg.svds(sm.asfptype(), k=start + decom_n * step, which='LM', tol=1e-4)
            U, S, V = U[:, ::-1], S[::-1], V.T[:, ::-1]
            print("lambdas:", S)
            kth  = k
            while kth < start + decom_n * step - 1 and kth < topk:
                if abs(max(U[:, kth])) < abs(min(U[:, kth])):
                    U[:, kth] *= -1
                if abs(max(V[:, kth])) < abs(min(V[:, kth])):
                    V[:, kth] *= -1
                row_cans = list(np.where(U[:, kth] >= 1.0 / np.sqrt(ms))[0])
                col_cans = list(np.where(V[:, kth] >= 1.0 / np.sqrt(ns))[0])
                if len(row_cans) <= 1 or len(col_cans) <= 1:
                    print("SKIP: candidates sizes are too small: {}".format((len(row_cans), len(col_cans))))
                    kth += 1
                    k += 1
                    continue
                sm_part = sm[row_cans, :][:, col_cans]
                row_ids, col_ids, avgsc_gs = greedy_func(sm_part, maxsize)
                
                kth += 1
                k += 1

                print("k_cur: {} size: {}, density: {}  @ {}s".format(kth, (len(row_ids), len(col_ids)), 
                                                                    avgsc_gs, time.time() - t1))
                if avgsc_gs > opt_density:
                    opt_k, opt_density = kth, avgsc_gs
                    (sm_pms, sm_pns) = sm_part.shape
                    fin_pms, fin_pns = len(row_ids), len(col_ids)
                    print("Update. svd tops shape (candidates size): {}".format((sm_pms, sm_pns)))
                    print("Update. size: {}, score: {}\n".format((fin_pms, fin_pns), avgsc_gs))

                    row_idx = dict(zip(range(sm_pms), sorted(row_cans)))
                    col_idx = dict(zip(range(sm_pns), sorted(col_cans)))
                    org_rownds = [row_idx[id] for id in row_ids]
                    org_colnds = [col_idx[id] for id in col_ids]
                    cans = [row_cans, col_cans]
                    orgnds = [org_rownds, org_colnds]
                    
                if 2.0 * opt_density >= S[kth]: # kth < topk and
                    print("Early Stopped. k_cur: {},  optimal density: {}, lambda_k: {}".format(kth, opt_density, S[kth]))
                    isbreak = True
                    break
            if isbreak:
                break
            decom_n += 1
        
        if orgnds is not None:
            print("\noptimal k: {}, density: {}".format(opt_k, opt_density))    
            print("total time @ {}s".format(time.time() - t1))
            return orgnds[0], orgnds[1], opt_density
        else:
            print("No dense subgraphs found.")
            return [], [], 0

    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass