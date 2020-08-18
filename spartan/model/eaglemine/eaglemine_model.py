#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

############################################################################
# Beyond outliers and on to micro-clusters: Vision-guided Anomaly Detection
# Authors: Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi,
#                 and Huawei Shen, and Xueqi Cheng
#
#  Project: eaglemine
#      eaglemine_model.py
#      Version:  1.0
#      Date: Dec. 17 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/17/2017>
#
#      Main contributor:  Wenjie Feng.
###########################################################################

__author__ = 'wenchieh'

# sys
import os
import operator
from collections import deque

# third-party lib
import numpy as np

# project
from .core.leveltree import LevelTree
from .utils.loader import Loader
from .utils.mdlbase import MDLBase
from .desc.dtmnorm_describe import DTMNormDescribe
from .desc.truncated_gaussian import TruncatedGaussian
from .desc.gaussian_describe import GaussianDescribe
from .desc.normal_gaussian import NormalGaussian
from .desc.statistic_hypothesis_test import StatisticHypothesisTest
from .desc.statistic_hypothesis_test_truncate import StatisticHypothesisTestTruncate


tiny_blobs = 'tiny_blob2cnt.out'
contracttree = 'level_tree_contract.out'
prunetree = 'level_tree_prune.out'
refinetree = 'level_tree_refine.out'


class EagleMineModel(object):
    _valid_vocabularies_ = ['dtmnorm', 'dmgauss']

    def __init__(self, mode=2, mix_comps=2, power_base=2):
        self.leveltree = None
        self.describes = None
        self.mode = mode
        self._hcel = None
        self._count = None
        self._total = None
        self.shape = None
        self.descvoc = None
        self._mixcomps = mix_comps
        self.power_base = power_base

    def set_vocabulary(self, desc_voc='dtmnorm'):
        # assert desc_voc in self._valid_vocabularies_
        if desc_voc == 'dtmnorm':
            self.descvoc = {"name": 'dtmnorm', "voc": DTMNormDescribe, "dist": TruncatedGaussian}
        elif desc_voc == 'dmgauss':
            self.descvoc = {"name": 'dmgauss', "voc": GaussianDescribe, "dist": NormalGaussian}
        else:
            raise ValueError("Unimplemented initialization method {}, "
                             "valid vocabularies {}".format(desc_voc, self._valid_vocabularies_))

    # def load_histogram(self, infn_hist):
    #     loader = Loader()
    #     shape, ticks, hist_arr = loader.load_histogram(infn_hist)
    #     self._hcel = hist_arr[:, :-1]
    #     self._count = hist_arr[:, -1]
    #     n_cel, n_dim = self._hcel.shape
    #     if n_dim != self.mode:
    #         raise ValueError("Input histogram dimension does match with the initial dimension.")
    #     self._total = n_cel
    #     self.shape = shape

    def set_histogram(self, hcel2cnt):
        self._hcel = np.asarray(hcel2cnt[:, :-1])
        self._count = np.asarray(hcel2cnt[:, -1])
        n_pos, n_dim = self._hcel.shape
        if n_dim != self.mode:
            raise ValueError("Input histogram dimension does match with the initial dimension.")
        self._total = n_pos
        self.shape = np.max(self._hcel, axis=0)

    def load_leveltree(self, infn, sep=',', verbose=True):
        self.leveltree = LevelTree()
        self.leveltree.load_leveltree(infn, sep, verbose)

    def load_describes(self, infn, sep=';', P=2, verbose=True):
        self.describes = dict()
        hcel2cnt = dict()
        for k in range(self._total):
            hcel2cnt[tuple(self._hcel[k])] = self._count[k]

        with open(infn, 'r') as ifp:
            for line in ifp:
                line = line.strip()
                splits = line.find(sep)
                node_id = int(line[:splits])
                desc = self.descvoc["voc"](self.mode)
                desc.load(line[splits+1:])
                tree_node = self.leveltree.get_node(node_id)
                node_counts = list()
                for hc in tree_node.get_covers():
                    node_counts.append(hcel2cnt.get(tuple(hc)))
                values = np.array(node_counts)
                values = np.log(values + 1) / np.log(P)
                desc.set_data(tree_node.get_covers(), values)
                self.describes[node_id] = desc
            ifp.close()

        if verbose:
            print('load describe done!')

    def dump(self):
        print("Information: ")
        print("#mode: {}, input-shape: {}".format(self.mode, self.shape))
        print("#non-zeros: {}, #totals: {}".format(len(self._hcel), np.sum(self._count)))
        print("describe vocabulary: {}".format(self.descvoc))
        print("Level-tree information:")
        self.leveltree.dump()
        print("Leaves describe information:")
        for id, desc in self.describes.items():
            print("{}: {}".format(id, str(desc)))

        #print("done!\n")

    def save(self, outfn, delimiter=";"): # , suffix=''
        # describe_outfn = 'describe%s.out' % suffix
        # os.path.join(outfd, describe_outfn)
        # self.leveltree.save_leveltree(outfd+'level_tree_expand-00.out')
        # print("save done!\n")
        with open(outfn, 'w') as ofp:
            desc_ids = sorted(self.describes.keys())
            for id in desc_ids:
                desc = self.describes[id]
                line = str(id) + delimiter + str(desc)
                ofp.writelines(line + '\n')
            ofp.close()

    def leveltree_build(self, outfd, step=0.2, prune_alpha=0.95, min_level=1.0, max_level=None, verbose=True):
        values = np.log2(1.0 + np.asarray(self._count, float))
        if max_level is None or max_level < min_level or max_level > np.max(values):
            max_level = np.max(values)

        print("\n========== ")
        print("Construct raw-tree.")
        self.leveltree = LevelTree()
        self.leveltree.build_level_tree(self._hcel, values, min_level, max_level, step,
                                        verbose=verbose, outfn = os.path.join(outfd, tiny_blobs))
        print("\n========== ")
        print("Refine tree structure.")
        print("\n+++++++++ ")
        print("a). tree contract")
        self.leveltree.tree_contract(verbose=verbose)
        if verbose:
            print("Info: Contract level-tree:")
            self.leveltree.dump()
        print("\n+++++++++ ")
        print("b). tree prune")
        self.leveltree.tree_prune(alpha=prune_alpha, verbose=verbose)
        if verbose:
            print("Info: Pruned level-tree:")
            self.leveltree.dump()
        print("\n+++++++++ ")
        print("c). tree node expand")
        self.leveltree.tree_node_expand(verbose)
        # self.leveltree.save_leveltree(os.path.join(outfd, refinetree))
        if verbose:
            print("Info: Expanded level-tree:")
            self.leveltree.dump()
        print("\ndone")

    def _describe_singular_check_(self, hcels):
        hcels = np.asarray(hcels)
        ndims = hcels.ndim
        sing_dims = list()
        for dm in range(ndims):
            if len(np.unique(hcels[:, dm])) <= 1:
                sing_dims.append(dm)

        if len(sing_dims) > 0:
            raise ValueError("{}-dimensional data degenerate in "
                             "{}-st dimension(s)".format(ndims, sing_dims))

    def describe_all(self):
        self.describes = dict()
        hcel2cnt = dict()
        for k in range(self._total):
            hcel2cnt[tuple(self._hcel[k])] = self._count[k]

        # tree nodes will be fitted with mixture distribution
        heavynode2level = self.leveltree.get_heavynodes()
        nodes = self.leveltree.get_nodes()  # self.leveltree.get_leaves()
        for ndid in nodes:
            tree_node = self.leveltree.get_node(ndid)
            hcels = tree_node.get_covers()
            node_counts = [hcel2cnt.get(tuple(hc)) for hc in hcels]
            is_mix = ndid in heavynode2level.keys()

            ## logP_count
            values = np.array(node_counts)
            values = np.log(values + 1) / np.log(self.power_base)

            comps = self._mixcomps if is_mix else 1
            desc = self.descvoc["voc"](self.mode, is_mix, comps)
            if self.descvoc["name"] == 'dtmnorm':
                desc.set_bounds()

            self._describe_singular_check_(hcels)
            desc.fit(hcels, values)
            self.describes[ndid] = desc

    def _model_hcels_prob_(self, hcel_left, hcel_right, paras, is_mix):
        npts = len(hcel_left)
        mus, covs, weights = paras["mus"], paras["covs"], paras["weights"]
        probs = None
        if is_mix:
            nmix = len(mus)
            mix_dists = list()
            for i in range(nmix):
                if self.descvoc["name"] == "dtmnorm":
                    desc = list(self.describes.values())[0]
                    lower_bnd, upper_bnd = desc.get_bounds()
                    idist = self.descvoc["dist"](lower_bnd, upper_bnd)
                else:
                    idist = self.descvoc["dist"]()
                idist.set_para(mus[i], covs[i])
                mix_dists.append(idist)

            comp_probs = list()
            for k in range(npts):
                ps = [mix_dists[i].range_cdf(hcel_left[k, :], hcel_right[k, :]) for i in range(nmix)]
                comp_probs.append(ps)
            probs = np.array(comp_probs)
        else:
            if self.descvoc["name"] == "dtmnorm":
                desc = list(self.describes.values())[0]
                lower_bnd, upper_bnd = desc.get_bounds()
                dist = self.descvoc["dist"](lower_bnd, upper_bnd)
            else:
                dist = self.descvoc["dist"]()
            dist.set_para(mus[0], covs[0])
            probs = np.array([dist.range_cdf(hcel_left[k, :], hcel_right[k, :]) for k in range(npts)])

        return probs

    def search(self, min_pts=20, strictness=4, verbose=True):
        search_tree = dict()
        blob_nodes = list()
        hcel2cnt = dict()
        min_pts = np.min([min_pts, int(np.mean(self._count))])
        for k in range(self._total):
            hcel2cnt[tuple(self._hcel[k])] = self._count[k]

        heavynode2level = self.leveltree.get_heavynodes()

        stat_tester = None
        if self.descvoc['name'] == 'dtmnorm':
            stat_tester = StatisticHypothesisTestTruncate(alpha_level=0.01, n_jobs=3)
        elif self.descvoc['name'] == 'dmgauss':
            stat_tester = StatisticHypothesisTest(strictness)
        else:
            raise ValueError("Unimplemented vocabulary, valid vocabularies {}".format(self._valid_vocabularies_))

        Q = deque()
        roots = self.leveltree.get_nodesid_atlevel(self.leveltree.levels[0])
        Q.extend(roots)

        # BFS search
        if self.describes is None:
            self.describes = dict()
        while len(Q) > 0:
            ndid = Q.popleft()
            tree_node = self.leveltree.get_node(ndid)
            hcels = np.array(tree_node.get_covers())
            node_counts = [hcel2cnt.get(tuple(hc)) for hc in hcels]

            if np.max(node_counts) < min_pts:
                blob_nodes.append(tree_node)

            is_mix = ndid in heavynode2level.keys()

            ## logP
            values = np.array(node_counts)
            values = np.log(values + 1) / np.log(self.power_base)

            self._describe_singular_check_(hcels)
            comps = self._mixcomps if is_mix else 1
            if ndid not in self.describes:
                desc = self.descvoc["voc"](self.mode, is_mix, comps)
                if self.descvoc["name"] == 'dtmnorm':
                    desc.set_bounds()
                desc.fit(hcels, values)
                self.describes[ndid] = desc
            else:
                desc = self.describes[ndid]
                desc.data, desc.values = hcels, values

            hcels_prob = None
            if is_mix:
                hcels_prob = self._model_hcels_prob_(hcels, hcels + 1, desc.paras, is_mix)

            weights = values #np.ones(len(hcels), int) #
            gaussian = False
            if self.descvoc['name'] == 'dmgauss':
                gaussian = stat_tester.apply(hcels, weights, hcels_prob, desc.paras, is_mix)
            elif self.descvoc['name'] == 'dtmnorm':
                lower_bnd, _ = desc.get_bounds()
                gaussian = stat_tester.apply(hcels, weights, hcels_prob, desc.paras, is_mix, lower_bnd)

            if gaussian:
                search_tree[ndid] = desc
                if verbose:
                    print("Info: island: {} hypothesis * Accept. ^~^".format(ndid))
            else:
                if tree_node.child is not None:
                    Q.extend(tree_node.child)
                    if verbose:
                        print("Info island: {} hypothesis & Rejected. >_<".format(ndid))
                else:
                    search_tree[ndid] = desc
                    if verbose:
                        print("Info: island: {} hypothesis $ pseudo-Accept. ^..^".format(ndid))

        self.describes = search_tree

    def _close_check(self, gaus_voc1, gaus_voc2, threshold=None):
        mu1, cov1 = np.array(gaus_voc1.get("mus")[0]), np.array(gaus_voc1.get("covs")[0])
        mu2, cov2 = np.array(gaus_voc2.get("mus")[0]), np.array(gaus_voc2.get("covs")[0])

        diff = mu1 - mu2 # distance between two cluster centers.
        distance = np.sqrt(diff.dot(diff.T))
        if threshold is not None:
            return distance <= threshold
        else:
            # default distance threshold
            cov_dist = np.max([1, np.max(np.sqrt(np.diag(cov1)) + np.sqrt(np.diag(cov2)))])
            return distance < 2 * cov_dist  #3 * np.sum(axes_distance)

    def _greedy_select(self, candidates, content):
        '''
        select the optimal merged cluster with max score (minimum decrease of log-likelihood)
        :param candidates: selection candidates
        :param clusters: all clusters dictionary
        :return:  merged cluster id
        '''
        if len(candidates) <= 0: return None

        cands_score = list()
        for cand in candidates:
            cs = content.get(cand)
            if len(cs.get("cs_id")) > 0:
                score = 0.0
                for cs_id in cs.get("cs_id"):
                    desc = content[cs_id].get("desc")
                    score += desc.paras["loss"]
                score -= cs.get("desc").paras["loss"]
                score /= 1.0 * cs.get("npts")            # average score over points
                # log-likelihood will decrease (need to select the one keep best model.)
                cands_score.append((cand, score))
        sorted_score = sorted(cands_score, cmp=lambda x, y: x[1]-y[1], reverse=False)
        return sorted_score[0][0]

    def post_stitch(self, strictness=4, verbose=True):
        optimals = list()
        mixturec, singlesc = None, dict()
        hcel2cnt = dict()
        for k in range(self._total):
            hcel2cnt[tuple(self._hcel[k])] = self._count[k]

        ider = -1
        for cid, desc in self.describes.items():
            if cid > ider: ider = cid
            if desc.is_mix: mixturec = cid
            else:
                npts = np.sum([hcel2cnt.get(tuple(pos)) for pos in desc.data])
                singlesc[cid] = {"id": cid, "desc": desc, "cs_id": [cid], "islnds": [cid], "npts": npts}
                optimals.append(cid)

        # stitching islands iteratively
        tested_cands = dict()

        stat_tester = None
        if self.descvoc['name'] == 'dtmnorm':
            stat_tester = StatisticHypothesisTestTruncate(alpha_level=0.05, n_jobs=3)
        elif self.descvoc['name'] == 'dmgauss':
            stat_tester = StatisticHypothesisTest(strictness)
        else:
            raise ValueError("Unimplemented vocabulary, valid vocabularies {}".format(self._valid_vocabularies_))

        while True:
            update = False
            candidates = list()
            n_cands = len(optimals)

            # test each clusters pair.
            for i in range(n_cands):
                ci = singlesc.get(optimals[i])
                for j in range(i + 1, n_cands):
                    cj = singlesc.get(optimals[j])
                    # have tested tuple
                    merge_tuple = (ci.get("id"), cj.get("id"))
                    merge_id = tested_cands.get(merge_tuple, None)
                    if verbose:
                        print(merge_tuple)
                    if merge_id is not None:
                        if merge_id != -1:
                            candidates.append(merge_id)
                        else:
                            continue
                    else:
                        desci, descj = ci.get("desc"), cj.get("desc")
                        tested_cands[merge_tuple] = -1
                        # merge limits: two islands are closed.
                        # distance = 0.3 * np.min(self.shape)
                        closed = self._close_check(desci.paras, descj.paras)
                        if not closed: continue

                        ## output testing cases
                        if verbose:
                            print("Info: test new merging node: {} ({}, {})".format(
                                merge_tuple, ci.get("cs_id"), cj.get("cs_id")))

                        merged_hcels = np.vstack([desci.data, descj.data])
                        merged_values = np.hstack([desci.values, descj.values])
                        desc = self.descvoc["voc"](self.mode, False, 1)
                        if self.descvoc["name"] == 'dtmnorm':
                            desc.set_bounds()
                        desc.fit(merged_hcels, merged_values)

                        weights = np.ones(len(merged_hcels), int) # merged_values #

                        gaussian = False
                        if self.descvoc['name'] == 'dmgauss':
                            gaussian = stat_tester.apply(merged_hcels, weights, None, desc.paras, False)
                        elif self.descvoc['name'] == 'dtmnorm':
                            lower_bnd, _ = desc.get_bounds()
                            gaussian = stat_tester.apply(merged_hcels, weights, None, desc.paras, False, lower_bnd)

                        if gaussian:
                            ider += 1
                            cs_id = [ci.get("id")] + [cj.get("id")]
                            islnds = list(set(ci.get("islnds") + cj.get("islnds")))
                            npts = ci.get("npts") + cj.get("npts")
                            singlesc[ider] = {"id": ider, "desc": desc, "cs_id": cs_id, "islnds": islnds, "npts": npts}
                            candidates.append(ider)
                            tested_cands[merge_tuple] = ider
                            update = True
                            if verbose:
                                print("new node: {}-{}-{}".format(ider, cs_id, islnds))
                        else:
                            if verbose:
                                print("Info:  ------ node {} check failed ------".format(merge_tuple))

            # select the most promising merge-node with greedy select from candidates
            #     (minimum decrease of log-likelihood)
            opt_merge_id = self._greedy_select(candidates, singlesc)
            if opt_merge_id is not None:
                update = True
                opt_c = singlesc.get(opt_merge_id)
                for cid in opt_c.get("cs_id"):
                    if cid in optimals:
                        optimals.remove(cid)
                    if cid in singlesc:
                        for islnd_id in singlesc.get(cid).get("islnds"):
                            if islnd_id in optimals:
                                optimals.remove(islnd_id)
                optimals.append(opt_merge_id)
                if verbose:
                    print("current optimals: {}".format(optimals))
            else:
                update = False

            if not update:
                break    # # no further update and stitch finished

        # if verbose:
        #     print("\nFinal optimal cluster result: {}".format(optimals))

        desc_stitch = dict()
        desc_stitch[mixturec] = self.describes.get(mixturec)
        for cid in optimals:
            c_nd = singlesc.get(cid)
            desc_stitch[cid] = c_nd.get("desc")
        self.describes = desc_stitch

    def _measure_model_mdl_(self, hcels, counts, hcels_label, hcels_prob, outlier_marker=-1):
        counts = np.asarray(counts)
        hcels_label = np.asarray(hcels_label)

        descs = list(self.describes.values())
        N_cls = len(descs)

        mdl = MDLBase()
        L_clusters = mdl.integer_mdl(N_cls)
        L_paras, L_assign, L_nis, L_error = 0.0, 0.0, 0.0, 0.0
        for k in range(N_cls):
            lb = k
            index = np.where(hcels_label == lb)
            k_hcels, k_cnts = hcels[index], counts[index]
            k_probs = hcels_prob[index]

            # model error code-length
            knis = np.sum(descs[k].values)
            exp_logPvs = k_probs * knis
            exp_counts = np.array(np.power(self.power_base, exp_logPvs), int) #exp_logPvs  #
            L_error += mdl.seq_diff_mdl(exp_counts, k_cnts)

            # model parameters code-length
            comp_parm = descs[k].compact_parm()
            L_paras += np.sum([mdl.float_mdl(p) for p in comp_parm])
            L_assign += 1.0  # for encoding the indicator for mixture or single
            L_nis += mdl.float_mdl(knis)  # for encoding #pts of this cluster

        L_outs = 0.0
        outs_index = np.where(hcels_label == outlier_marker)
        outs_hcels, outs_vals = hcels[outs_index], counts[outs_index]
        L_outs += mdl.seq_diff_mdl(np.zeros_like(outs_vals, int), outs_vals)
        L_shape = np.sum([mdl.integer_mdl(dm) for dm in self.shape])

        C = L_clusters + L_assign + L_shape + L_paras + L_nis + L_outs + L_error
        return C

    def _smooth_entropy_(self, ps, qs, weights, smoother=1e-15):
        n = len(ps)
        ps = np.asarray(ps) + smoother
        qs = np.asarray(qs) + smoother
        w_en = [ weights[i] * ps[i] * np.log2(ps[i] * 1.0 / qs[i])
                 if qs[i] != 0 else 0 for i in range(n) ]
        return np.sum(w_en)

    def _measure_suspicious_(self, hcels, values):
        majority_normal = None
        cls_id2prob = dict()
        for id, desc in self.describes.items():
            hcel_prob = self._model_hcels_prob_(hcels, hcels + 1, desc.paras, desc.is_mix)
            if desc.is_mix:
                majority_normal = id
                hcel_prob = np.sum(hcel_prob * desc.paras['weights'], axis=1)
            cls_id2prob[id] = hcel_prob

        dists = dict()
        for id, probs in cls_id2prob.items():
            dists[id] = self._smooth_entropy_(probs, cls_id2prob[majority_normal], values)

        return sorted(dists.items(), key=operator.itemgetter(1), reverse=True)

    def _hcels_labeling_(self, hcels, outliers_marker=-1, strictness=4):
        criticals = [1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5]
        hcels = np.asarray(hcels)
        hcel_clsprob = list()
        hcel_labs = list()

        descs = list(self.describes.values())
        mixId = -1
        for k in range(len(descs)):
            desc = descs[k]
            clsprob = self._model_hcels_prob_(hcels, hcels + 1, desc.paras, desc.is_mix)
            if desc.is_mix:
                mixId = k
                clsprob = np.sum(clsprob * desc.paras['weights'], axis = 1)
            hcel_clsprob.append(clsprob)
        hcel_clsprob = np.array(hcel_clsprob).T

        for k in range(self._total):
            prob, cls = np.max(hcel_clsprob[k, :]), np.argmax(hcel_clsprob[k, :])
            if prob <= criticals[strictness]:
                label = outliers_marker
            else:
                label = cls

            hcel_labs.append(label)

        probs = np.max(hcel_clsprob, axis=1)
        return hcel_labs, probs
    
    def cluster_remarks(self, strictness=4, verbose=True):
        outliers_marker = -1
        hcels_labs, hcels_prob = self._hcels_labeling_(self._hcel, outliers_marker, strictness)
        mdls = self._measure_model_mdl_(self._hcel, self._count, hcels_labs, hcels_prob, outliers_marker)
        hcel2label = dict(zip(map(tuple, self._hcel), hcels_labs))
        # suspicious = self._measure_suspicious_(self._hcel, np.log2(1 + self._count))
        if verbose:
            print("** mdl (Model Description Length) for describing the graph feature: {} **".format(mdls))
            # print("Island suspicious orders: {}".format(suspicious))
        
        return hcel2label, mdls, #suspicious
    
    def cluster_histogram(self, outfn=None, strictness=4, comments="#", delimiter=",", verbose=True):
        outliers_marker = -1
        hcels_label, hcels_prob = self._hcels_labeling_(self._hcel, outliers_marker, strictness)
        mdls = self._measure_model_mdl_(self._hcel, self._count, hcels_label, hcels_prob, outliers_marker)
        # suspicious = self._measure_suspicious_(self._hpos, np.log2(1 + self._count))

        if outfn is not None:
            np.savetxt(outfn, np.vstack((self._hcel.T, hcels_label)).T, '%i', delimiter=delimiter, comments=comments,
                       header='#hcel: {}, #label: {}'.format(self._total, len(np.unique(hcels_label))))
        if verbose:
            print("** mdl (Model Description Length) for describing the graph feature: {} **".format(mdls))
            # print("Island suspicious orders: {}".format(suspicious))

        return self._hcel, hcels_label, mdls
        
    def graph_node_cluster(self, node2hcel, outfn_node2lab, outfn_hcel2lab=None, 
                           strictness=4, comments="#", sep=","):
        
        outliers_marker = -1
        hcls_labs, _ = self._hcels_labeling_(self._hcel, outliers_marker, strictness)
        hcel2lab = dict(zip(map(tuple, self._hcel), hcls_labs))
        
        with open(outfn_node2lab, 'w') as ofp:
            ofp.writelines(comments + " #pt: {}, #label: {}\n".format(len(node2hcel), len(np.unique(hcls_labs))))
            for k in range(len(node2hcel)):
                ndidx, pos = node2hcel[k, 0], tuple(node2hcel[k, 1:])
                ndlab = hcel2lab.get(pos, -1)
                ofp.writelines("{}{}{}\n".format(ndidx, sep, ndlab))
            ofp.close()
        
        if outfn_hcel2lab is not None:
            np.savetxt(outfn_hcel2lab, np.vstack((self._hcel.T, hcls_labs)).T, '%i', delimiter=sep, comments=comments,
                       header='#hcel: {}, #label: {}'.format(self._total, len(np.unique(hcls_labs))))
        
    def cluster_weighted_suspicious(self, hcel_weights, strictness=4, verbose=True):
        outliers_marker = -1
        hcels_lab, hcels_prob = self._hcels_labeling_(self._hcel, outliers_marker, strictness)
        _susp_ = self._measure_suspicious_(self._hcel, np.log2(1 + self._count))

        cls2susp = dict(_susp_)
        cls2wtsusp = dict()

        hcels_lab = np.asarray(hcels_lab)
        clsid2labs = dict(zip(self.describes.keys(), range(len(self.describes.values()))))
        for clsid, desc in self.describes.items():
            lb = clsid2labs[clsid]
            lb_hcels_idx = np.arange(len(hcels_lab))[hcels_lab == lb]
            cls_wt, npts = 0, 0
            for hidx in lb_hcels_idx:
                cls_wt += hcel_weights[hidx] * self._count[hidx]
                npts += self._count[hidx]
            cls_wt /= 1.0 * npts
            cls2wtsusp[clsid] = cls2susp[clsid] * np.log(cls_wt)

        sorted_wtsusp = sorted(cls2wtsusp.items(), key=operator.itemgetter(1), reverse=True)

        if verbose:
            _suspstr_ = ", ".join(["(%d, %.2f)"%(d[0], d[1]) for d in _susp_])
            print("kl-divergence based suspicious: [{}]".format(_suspstr_))
            _wtsuspstr_ = ", ".join(["(%d, %.2f)"%(d[0], d[1]) for d in sorted_wtsusp])
            print("weighted suspicious orders: [{}]".format(_wtsuspstr_))

        return sorted_wtsusp

