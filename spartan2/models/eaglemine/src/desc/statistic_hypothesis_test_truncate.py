#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Statistic hypothesis Test for truncate normal distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      statistic_hypothesis_test_truncate.py
#      Version:  1.0
#      Date: December 1 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/1/2017>
#

__author__ = 'wenchieh'

# third-party lib
import numpy as np

# project
from .truncate_norm_gof import TruncatedNormalGoF


class StatisticHypothesisTestTruncate(object):
    def __init__(self, alpha_level=0.1, n_jobs=1):
        self.alpha = alpha_level
        self._n_jobs_ = n_jobs
        self._min_float = np.finfo('float64').resolution

    def _expand_(self, xs, count):
        if np.max(count) == 1:
            return np.array(xs)
        else:
            exp_xs = list()
            for k in range(len(xs)):
                exp_xs.extend([list(xs[k])] * count[k])
            return np.asarray(exp_xs)

    def _mixture_assign_prob_(self, xs, xs_probs, components, mix_weights):
        assigns = list()
        xs_wprob = np.array(xs_probs) * np.array(mix_weights)
        xs_comp_ = np.argmax(xs_wprob, axis=1)
        for k in range(components):
            assigns.append(np.where(xs_comp_ == k)[0])

        return assigns

    def _trunc_gaussian_check_(self, ndim, xs, means, stds, Hs, verbose=False):
        trunc_gaus = True
        gofer = TruncatedNormalGoF(self._n_jobs_)
        for k in range(ndim):
            # k_res = gofer.ad2_test(xs[:, k], means[k], stds[k], Hs[k])
            k_res = gofer.ad2up_test(xs[:, k], means[k], stds[k], Hs[k])
            # k_res = gofer.adup_test(xs[:, k], means[k], stds[k], Hs[k], 'greater')
            k_TS, k_pvalue = k_res['statistic'], k_res['p_value']
            if verbose:
                print("dim: {}, TS: {}, p-value:{}".format(k, k_TS, k_pvalue))
            if k_pvalue < self.alpha:
                trunc_gaus = False
                break
        return trunc_gaus

    def apply(self, xs, xs_weights, xs_probs,  paras, is_mix, Hs):
        xs = np.asarray(xs)
        xs = self._expand_(xs, np.array(np.ceil(xs_weights), int))
        m, n = xs.shape
        if is_mix:
            mus, covs, mix_wt = paras.get("mus"), paras.get("covs"), paras.get("weights")
            n_mixs = len(mus)
            xs_probs = self._expand_(xs_probs, np.array(np.ceil(xs_weights), int))
            assigns = self._mixture_assign_prob_(xs, xs_probs, n_mixs, mix_wt)
            mix_trunc_gaus = True
            for i in range(n_mixs):
                xs_imix = xs[assigns[i], :]
                mix_trunc_gaus &= self._trunc_gaussian_check_(n, xs_imix, mus[i],
                                                              np.sqrt(np.diag(covs[i])), Hs)
                if not mix_trunc_gaus:
                    break
            return mix_trunc_gaus

        else:
            means, stds = paras.get("mus")[0], np.sqrt(np.diag(paras.get("covs")[0]))
            return self._trunc_gaussian_check_(n, xs, means, stds, Hs)