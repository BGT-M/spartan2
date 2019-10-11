#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  (Discrete) Truncated Gaussian distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      truncated_gaussian.py
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

# third-party lib
import numpy as np
from numpy.linalg import det, inv

# project
from .norm_extras import mvnormcdf

class TruncatedGaussian(object):
    def __init__(self, lower_bound, upper_bound):
        self._ndim = 0
        self._mu, self._cov = None, None
        self._lower, self._upper = np.array(lower_bound), np.array(upper_bound)

    def set_para(self, mu, cov):
        self._mu = np.array(mu)
        self._ndim = len(mu)
        self._cov = np.array(cov)

    def _phi_(self, xs, mu, cov):
        nc = 2.0 * np.pi * np.sqrt(det(cov))
        norm_vect = np.exp(-1 * (xs - mu).dot(inv(cov)).dot((xs - mu).transpose()) / 2.0) / nc
        return norm_vect

    def _log_phi_(self, xs, mu, cov):
        return np.log(self._phi_(xs, mu, cov))

    def _psi_(self, xs, mu, cov, lower, upper):
        for k in range(self._ndim):
            if xs[k] < lower[k] or xs[k] > upper[k]:
                return 0

        p_xy = self._phi_(xs, mu, cov)
        normalizer = mvnormcdf(upper, mu, cov, lower)
        return p_xy * 1.0 / normalizer

    def _log_psi_(self, xs, mu, cov, lower, upper):
        return np.log(self._psi_(xs, mu, cov, lower, upper))

    def pdf(self, xs):
        ''' the pdf of xs in truncate normal distribution'''
        return self._psi_(xs, self._mu, self._cov, self._lower, self._upper)

    def cdf(self, xs):
        ''' the probability of xs in truncate normal distribution. given mu, Sigma, and truncated lower bound.'''
        for k in range(self._ndim):
            if xs[k] < self._lower[k] or xs[k] > self._upper[k]:
                return 0

        xs_upper = np.array(xs)
        return mvnormcdf(xs_upper, self._mu, self._cov, self._lower)

    def range_cdf(self, left, right):
        for k in range(self._ndim):
            if left[k] < self._lower[k] or right[k] > self._upper[k]:
                return 0

        denorm = 1.0 * mvnormcdf(self._upper, self._mu, self._cov, self._lower)
        if denorm == 0:
            return 0
        p = 1.0 * mvnormcdf(right, self._mu, self._cov, left)
        return np.max([0,  p / denorm])