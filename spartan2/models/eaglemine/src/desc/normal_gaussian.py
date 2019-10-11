#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  (Discrete) Normal Gaussian distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      normal_gaussian.py
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
from scipy.stats import multivariate_normal

# project
from .norm_extras import mvnormcdf


class NormalGaussian(object):
    def __init__(self):
        self._mu, self._cov = None, None
        self._ndim = 0

    def set_para(self, mu, cov):
        self._mu = np.array(mu)
        self._ndim = len(mu)
        self._cov = np.array(cov)

    def pdf(self, xs):
        return multivariate_normal.pdf(xs, self._mu, self._cov, allow_singular=True)

    def cdf(self, xs):
        return mvnormcdf(xs, self._mu, self._cov)

    def range_cdf(self, left, right):
        return mvnormcdf(right, self._mu, self._cov, left)