#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Histogram generator with heuristic rules for eliminating black degree lines for degree sequence.
#  Author: wenchieh
#
#  Project: eaglemine
#      histogram_heuristic_generator.py
#      Version:  1.0
#      Date: December 17 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/17/2017>
# 

__author__ = 'wenchieh'

# sys
import bisect

# third-party lib
import numpy as np


class HistogramHeuristicGenerator(object):
    def __init__(self):
        self._mode = 0
        self._npts = 0
        self._features = None
        self._ticks_vec = None
        self.shape = None
        self.histogram = None
        self.points_coord = None
        self.hpos2avgfeat = None

    def set_deg_data(self, degrees, features):
        assert(len(degrees) == len(features))
        # self._features = np.hstack([np.array(degrees).T[:, np.newaxis], np.asarray(features).T[:, np.newaxis]])
        self._features = np.column_stack((degrees, features))
        self._npts, self._mode = self._features.shape

    def set_data(self, features):
        self._features = np.array(features)
        self._npts, self._mode = self._features.shape

    def _bucketize_N_(self, values, N, logarithmic=False, base=10):
        values = np.array(values)
        if logarithmic:
            if values.min() < 0:
                values = values[values >= 0]
            if values.min() == 0:
                values += 1.0
            values = np.log(values) / np.log(base)

        min_val, max_val = values.min(), values.max()
        bandwidth = (max_val - min_val) * 1.0 / N
        bins_vec = np.array([min_val + i * bandwidth for i in range(N + 1)])

        if logarithmic:
            bins_vec = np.power(base, bins_vec)

        return bins_vec

    def _bucketize_logP_(self, values, P):
        logP = np.log(P)
        values = np.array(values)
        values = values[values > 0]
        min_lgpv, max_lgpv = np.log(values.min()) * 1.0 / logP, np.log(values.max()) * 1.0 / logP
        N_bins = max_lgpv - min_lgpv + 1
        bins_vec = np.array(np.power(P, min_lgpv + i * logP) for i in range(N_bins))

        return bins_vec

    def _bucketize_degree_(self, degrees, base=10):
        start, step = 0, 0
        if base == 10:
            start, step = 10, 0.05
        elif base == 2:
            start, step = 16, 0.1
        else:
            print("Error exception: Not a proper step setting")
            exit()

        sparse_seq = range(1, start + 1)
        min_lgdeg = np.log(np.max(sparse_seq)) * 1.0 / np.log(base)
        max_lgdeg = np.log(np.max(degrees + 1)) * 1.0 / np.log(base)
        init_dense_seq = np.power(base, np.arange(min_lgdeg + step, max_lgdeg + step, step))
        init_dense_seq = np.asarray(init_dense_seq, int)
        dense_seq = list()
        dense_seq.extend(init_dense_seq[:2])

        for i in range(2, len(init_dense_seq)):
            pre_bw = int(dense_seq[-1] - dense_seq[-2])
            cur = init_dense_seq[i]
            if int(cur - dense_seq[-1]) < pre_bw:
                dense_seq.append(dense_seq[-1] + pre_bw)
            else:
                dense_seq.append(cur)

        bins_vec = np.hstack([sparse_seq, np.asarray(dense_seq)])

        return bins_vec

    def _seq2hists_(self, seq, ticks):
        Npts, nbins = len(seq), len(ticks) - 1
        N_outliers = 0

        hist = np.array([0] * nbins, int)
        pts_pos = np.array([0] * Npts, int)

        for pt_idx in range(Npts):
            pt_val = seq[pt_idx]
            if pt_val < ticks[0]:
                N_outliers += 1
                pts_pos[pt_idx] = -1
            else:
                k_idx = bisect.bisect_left(ticks, pt_val)
                k_idx = min([k_idx, nbins])
                if k_idx > 0:
                    if abs(ticks[k_idx] - pt_val) > 1e-14:
                        k_idx = max([0, k_idx - 1])
                    k_idx = min([k_idx, nbins - 1])
                hist[k_idx] += 1
                pts_pos[pt_idx] = k_idx

        return hist, pts_pos, N_outliers

    def _check_data_(self):
        if np.min(self._features) <= 0.0:
            ValueError("Error: The input feature contains non-positive values, which can't be applied with logarithmic")

    def _degree_feature_bucketize_(self, base):
        if self._ticks_vec is None:
            self._ticks_vec = list()
        if self.shape is None:
            self.shape = list()

        deg_bucks = self._bucketize_degree_(self._features[:, 0], base)
        self._ticks_vec.append(deg_bucks)
        self.shape.append(len(deg_bucks))
        for mod in range(1, self._mode):
            values = self._features[:, mod]
            values = values[values > 0]
            min_lgpv, max_lgpv = np.log(values.min()) * 1.0 / np.log(base), np.log(values.max()) * 1.0 / np.log(base)
            N_bins = int(np.ceil(len(deg_bucks) / 10.0)) * 10
            bucks = min_lgpv + (max_lgpv - min_lgpv) * np.arange(N_bins + 1) * 1.0 / N_bins
            self._ticks_vec.append(np.power(base, bucks))
            self.shape.append(len(bucks))

    def _multidiscrete_bucketize_(self, discretes_index, base):
        '''
        bucketize for multiple discrete features in log scale with base=base, like degree, # triangle, etc.
        :param discretes_index: discrete features index in data
        :param base: the base of log
        '''
        if self._ticks_vec is None:
            self._ticks_vec = list()
        if self.shape is None:
            self.shape = list()

        max_disc_bins = 0
        for mod in range(self._mode):
            values = self._features[:, mod]
            values = values[values > 0]
            if mod in discretes_index:
                disc_bucks = self._bucketize_degree_(values, base)
                self._ticks_vec.append(disc_bucks)
                self.shape.append(len(disc_bucks))
                if len(disc_bucks) > max_disc_bins:
                    max_disc_bins = len(disc_bucks)
            else:
                min_lgpv = np.log(values.min()) * 1.0 / np.log(base)
                max_lgpv = np.log(values.max()) * 1.0 / np.log(base)
                N_bins = int(np.ceil(max_disc_bins / 10.0)) * 10
                bucks = min_lgpv + (max_lgpv - min_lgpv) * np.arange(N_bins + 1) * 1.0 / N_bins
                self._ticks_vec.append(np.power(base, bucks))
                self.shape.append(len(bucks))

    def histogram_gen(self, method="degree", **argv):
        """
        histogram generator with heuristic strategies specified with parameter [method]
        :param method: specify the histogram strategy:  "degree" | "mdisc" | "N" | "logP"
        :param argv:  variable argument for specified strategy.
                method   |   argv
                degree   |   base (default=10)
                mdisc    |   discindex   a list indicating the discrete feature index
                         |   base  (default=10)
                N        |   N (or N(s) a list for each dimension feature)
                         |   logarithmic (default=False), 
                         |   base(default=10, if logarithmic is True)
                logP     |   P (the log-base) (or P(s) a list for each dimension feature)
        :return:
        """
        heuristic = ["degree", "N", "logP"]
        if method not in heuristic:
            ValueError("Parameter Error: heuristic strategy should be in {}".format(heuristic))
            exit()

        self.shape = list()
        self._ticks_vec = list()
        if method is "degree":
            # N = 100
            base = 10
            if argv.get("N", None) is not None:
                N = 100
            if argv.get("base", None) is not None:
                base = argv["base"]
            self._check_data_()
            self._degree_feature_bucketize_(base)
        elif method is 'mdisc':
            base = 10
            mdisc_index = None
            if argv.get("mdisc_index", None) is not None:
                mdisc_index = argv["mdisc_index"]
            if argv.get("base", None) is not None:
                base = argv["base"]
            self._check_data_()
            self._multidiscrete_bucketize_(mdisc_index, base)
        elif method is "N":
            N, logarithmic, base = argv.get("N"), argv.get("logarithmic", False), argv.get("base", 10)
            Ns = list()
            if isinstance(N, int):   Ns = list([N]) * self._mode
            elif isinstance(N, list):  Ns = list(N)
            else:
                print("Parameter Error: N should be an integer or a list for bucketize the features")
                exit()

            if logarithmic is True:
                self._check_data_()

            for mod in range(self._mode):
                bucks = self._bucketize_N_(self._features[:, mod], Ns[mod], logarithmic, base)
                self._ticks_vec.append(bucks)
                self.shape.append(len(bucks))
        else:
            self._check_data_()
            P = argv.get("P")
            Ps = list()
            if isinstance(P, int):  Ps = list([P]) * self._mode
            elif isinstance(P, list):  Ps = list(P)
            else:
                print("Parameter Error: P should be an integer or a list for bucketize the features")
                exit()

            for mod in range(self._mode):
                bucks = self._bucketize_logP_(self._features[:, mod], Ps[mod])
                self._ticks_vec.append(bucks)
                self.shape.append(len(bucks))

        points_coord = list()
        for mod in range(self._mode):
            _, pts_pos, _ = self._seq2hists_(self._features[:, mod], self._ticks_vec[mod])
            pts_pos[pts_pos < 0] = 0
            points_coord.append(pts_pos)

        self.points_coord = np.asarray(points_coord, int).T
        self.histogram = dict()
        for pos in self.points_coord:
            self.histogram[tuple(pos)] = self.histogram.get(tuple(pos), 0) + 1

        self.hpos2avgfeat = dict()
        for i in range(self._npts):
            pos, feat = tuple(self.points_coord[i]), self._features[i]
            self.hpos2avgfeat[pos] = self.hpos2avgfeat.get(pos, np.zeros_like(feat)) + feat

        for pos in self.hpos2avgfeat.keys():
            self.hpos2avgfeat[pos] /= 1.0 * self.histogram[pos]

    def save_histogram(self, outfn, sep=',', comments="#"):
        with open(outfn, 'w') as ofp:
            header = comments + " {}".format(str(self.shape)[1:-1].replace(', ', sep))
            ofp.writelines(header + "\n")
            for vec in self._ticks_vec:
                ticks_str = sep.join(map(str, vec))
                ofp.writelines("# " + ticks_str + "\n")

            for pos, cnt in self.histogram.items():
                pos_str = sep.join(map(str, pos))
                line = "{}{}{}".format(pos_str, sep, cnt)
                ofp.writelines(line + "\n")
            ofp.close()

    def save_pts_index(self, outfn, sep=',', comments="#", pts_idx=None):
        header = comments + "pts: {}; ranges: {}".format(len(self._features), str(self.shape)[1:-1].replace(', ', sep))
        index = np.arange(self._npts) if pts_idx is None else np.asarray(pts_idx, int)
        out_dt = np.vstack([index, self.points_coord.T]).T
        np.savetxt(outfn, out_dt, '%d', sep, header=header)
        # print("save points2pos done!")

    def save_hpos2avgfeat(self, outfn, sep=',', comments="#"):
        with open(outfn, 'w') as ofp:
            pos_xs, feat_vs = "", ""
            for k in range(self._mode):
                pos_xs += "hpos-{}{}".format(k+1, sep)
                feat_vs += "avgfeat-{}{}".format(k+1, sep)
            headers = pos_xs + feat_vs + "#npts"
            ofp.writelines(comments + " " + headers + '\n')
            for pos, feat in self.hpos2avgfeat.items():
                pos_str = sep.join(map(str, pos))
                feat_str = str(feat.tolist())[1:-1].replace(', ', sep)
                line = pos_str + sep + feat_str + sep + str(self.histogram[pos])
                ofp.writelines(line + '\n')
            ofp.close()

    def dump(self):
        print("Histogram Info:")
        print("\t Histogram shape: {}".format(self.shape))
        print("\t #points: {}, #mode: {}".format(self._npts, self._mode,))
