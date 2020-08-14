#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Statistic hypothesis test class
#  Author: wenchieh
#
#  Project: eaglemine
#      statistic_hypothesis_test.py
#      Version:  1.0
#      Date: December 05 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/5/2017>
#

__author__ = 'wenchieh'

# third-party lib
import numpy as np
from numpy.linalg import eig
from sklearn.preprocessing import scale
from scipy.stats import anderson

# project
from .norm_extras import norm_anderson


class StatisticHypothesisTest(object):
    def __init__(self, strictness=4):
        self.strict = strictness
        self._min_float = np.finfo('float64').resolution

    def _expand_(self, xs, count):
        if np.max(count) == 1:
            return np.array(xs)
        else:
            exp_xs = list()
            for k in range(len(xs)):
                exp_xs.extend([list(xs[k])] * count[k])
            return np.asarray(exp_xs)

    def _gaussian_anderson_(self, vector, strictness):
        """
        check whether a given input vector follows a gaussian distribution
        H0: vector is distributed gaussian
        H1: vector is not distributed gaussian
        :param vector [array], all_data to be checked
        :param strictness [int default=4], {0, 1, 2, 3, 4}
                     how strict should the anderson-darling test for normality be
                     0: not at all strict;     4: very strict
             Critical values provided are for the following significance levels:
               normal/exponential:     15%, 10%, 5%, 2.5%, 1%
        """
        output = anderson(vector)
        if output[0] <= output[1][strictness]:
            return True
        else:
            return False

    def _gaussian_check(self, xs ,cov):
        """
        project data [xs] into eigenvectors specific axises to be one-dimensional for statistical test
        References
        ----------
         [1] Hamerly, Greg and Charles Elkan. “Learning the k in k-means.” NIPS (2003).
        ----------
        """
        _, v = eig(np.asarray(cov))
        gaus = True
        for k in range(len(cov)):
            kv = v[:, k]
            proj_kvs = scale(xs.dot(kv) / (kv.dot(kv)) + self._min_float)
            gaus &= self._gaussian_anderson_(proj_kvs, self.strict)
            if not gaus:
                break

        return gaus

    def _mixture_assign_scale_(self, xs, mus, covs, scale=5):
        assigns = list()
        nxs = len(xs)
        components, dims = np.array(covs).shape

        for i in range(components):
            mu, cov = mus[i], covs[i]
            _, v = eig(np.asarray(cov))
            belongs = np.array([True] * nxs)
            for k in range(dims):
                kv = v[:, k]
                sigma = np.sqrt(cov[k, k])
                proj_kv = abs((xs - mu).dot(kv) / (kv.dot(kv)))
                belongs &= (proj_kv <= scale * sigma)
            assigns.append(xs[belongs, :])

    def _mixture_assign_prob_(self, xs, xs_probs, components, mix_weights):
        assigns = list()
        xs_wprob = np.array(xs_probs) * np.array(mix_weights)
        xs_comp_ = np.argmax(xs_wprob, axis=1)
        for k in range(components):
            assigns.append(np.where(xs_comp_ == k)[0])

        return assigns

    def apply(self, xs, xs_weights, xs_probs,  paras, is_mix, argv=None):
        xs = np.asarray(xs)
        xs = self._expand_(xs, np.array(np.ceil(xs_weights), int))

        if is_mix:
            mus, covs, mix_wt = paras.get("mus"), paras.get("covs"), paras.get("weights")
            n_mixs = len(mus)
            xs_probs = self._expand_(xs_probs, np.array(np.ceil(xs_weights), int))
            assigns = self._mixture_assign_prob_(xs, xs_probs, n_mixs, mix_wt)
            ## assigns = self._mixture_assign_scale_(xs, mus, covs, scale=5)
            mix_gaus = True
            for i in range(n_mixs):
                xs_imix = xs[assigns[i], :]
                mix_gaus &= self._gaussian_check(xs_imix, covs[i])
                if not mix_gaus:
                    break
            return mix_gaus
        else:
            cov = paras.get("covs")[0]
            return self._gaussian_check(xs, cov)
