#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Describe hypercubes (in high dimensional) with DTM Gaussian distribution
#      DTM Gaussian: Discrete Truncate Multivariate Normal Gaussian.
#  Author: wenchieh
#
#  Project: eaglemine
#      dtmnorm.py
#      Version:  1.0
#      Date: December 1 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/01/2017>
#
__author__ = 'wenchieh'

# sys
import collections

# third-party lib
import numpy as np
from numpy import inf
from numpy.linalg import det
import scipy.optimize as opt
from statsmodels.stats.weightstats import DescrStatsW
from sklearn import cluster
from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution

# project
from .norm_extras import mvnormcdf


class DTMNorm(object):
    def __init__(self, ndim=2, lower_bound=None, upper_bound=None):
        self.ndim = ndim
        if lower_bound is None:
            lower_bound = np.array([0] * ndim)
        self.lowers = np.array(lower_bound)
        if upper_bound is None:
            upper_bound = np.array([inf] * ndim)
        self.uppers = np.array(upper_bound)

    def set_lowers(self, lower_bound):
        self.lowers = np.array(lower_bound)

    def get_lowers(self):
        return self.lowers

    def set_uppers(self, upper_bound):
        self.uppers = np.array(upper_bound)

    def get_uppers(self):
        return self.uppers

    def _paras_compose_(self, mus, covs, ws):
        paras = list()
        paras.extend(list(np.array(mus).flatten()))

        for i in range(len(covs)):
            cov = covs[i]
            for k in range(self.ndim):
                paras.extend(np.array(cov[k, k:]).tolist())

        if ws is not None:
            paras.extend(ws[:-1])
        return np.array(paras)

    def _paras_decompose_(self, paras, n_components):
        mus, covs, ws = list(), list(), list()
        mus = list(np.array(paras[: self.ndim * n_components]).reshape((n_components, self.ndim)))
        covupt_len = range(self.ndim, 0, -1)
        for i in range(n_components):
            cov_upt = np.zeros((self.ndim, self.ndim))
            covs_starts = self.ndim * n_components + i * np.sum(covupt_len)
            for k in range(self.ndim):
                start = int(covs_starts + np.sum(covupt_len[:k]))
                end = int(covs_starts + np.sum(covupt_len[:k+1]))
                cov_upt[k, k:] = np.array(paras[start: end])
            cov = cov_upt + cov_upt.T - np.diag(np.diag(cov_upt))
            covs.append(cov)

        mc_parm_len = len(mus) + n_components * np.sum(covupt_len)
        if mc_parm_len < len(paras):
            ws = list(paras[-(n_components-1):])
            ws = ws + list([1.0 - np.sum(ws)])
        else:
            ws = [1.0 / n_components] * n_components
        return mus, covs, np.asarray(ws)

    #TODO: non-positive covariance matrix process [refinement]
    def _cov_process_(self, cov, reg_covar=1e-4):
        '''
        process the un-expected case: cov is non-positive definitive covariance
            only use diagonal matrix of cov to replace current cov.
        other suggestions:
               reference to:
                   Not Positive Definite Matrices--Causes and Cures
                   [http://www2.gsu.edu/~mkteer/npdmatri.html]
        '''
        ## Non-negative regularization added to the diagonal of covariance.
        ## Allows to assure that the covariance matrices are all positive.
        if det(cov) <= 0.0:
            diags = np.diag(cov)
            if 0.0 in diags:
                diags += reg_covar
            cov = np.diag(diags)
            # diag_min = np.min(diags)
            # if diag_min < 0:
            #     for k in range(self.ndim):
            #         cov[k, k] -= diag_min
        return cov

    def _single_optpara(self, paras_vec, pos_left, pos_right, weights, debug_info=None):
        res = 0.0
        N = len(pos_left)
        mus, covs, _ = self._paras_decompose_(paras_vec, 1)
        cov = self._cov_process_(covs[0])

        # smoothed normalizer
        normalizer = np.max([8e-3, mvnormcdf(self.uppers, mus[0], cov, self.lowers)])
        for i in range(N):
            numerator = np.max([0, mvnormcdf(pos_right[i, :], mus[0], cov, pos_left[i, :])])
            P_pos = numerator / normalizer
            if P_pos > 0:
                res += weights[i] * np.log(P_pos)
            else:
                res += -1.0 * np.NaN

        if debug_info is not None:
            debug_info.append(-res)
            if len(debug_info) % 200  == 0:
                print(len(debug_info), -res)

        return -1 * res

    def fit_single(self, pos_left, pos_right, weights, tol=1e-4, maxiter=4000, verbose=False):
        left, right = np.asarray(pos_left), np.asarray(pos_right)
        debugs = list() if verbose else None
        centers = (left + right) / 2.0
        statsW = DescrStatsW(centers, weights=np.array(weights))
        init_paras = self._paras_compose_([statsW.mean], [statsW.cov], [1.0])

        method = 'Nelder-Mead'
        res = opt.minimize(self._single_optpara, init_paras, args=(left, right, weights, debugs),
                           method = method, tol = tol, options={'maxiter': maxiter, 'disp': verbose})
        if verbose:
            print("Method:{}; Initial parameter: {};".format(method, init_paras))
            print("Converged Parameter: {}".format(res.x))

        mus, covs, ws = self._paras_decompose_(res.x, 1)
        if det(covs[0]) == 0.0:
            print("Warning: covariance processed:")
            print("\t pre-optimal mus: {}, cov: {}".format(mus[0], covs[0]))
            covs[0] = self._cov_process_(covs[0])

        return mus, covs, ws, res.fun

    def _mixture_optpara(self, paras_vec, pos_left, pos_right, weights, n_components=2, debug_info=None):
        res = 0.0
        N = len(pos_left)
        mus, covs, comp_ws = self._paras_decompose_(paras_vec, n_components)
        normalizors = list()
        for k in range(n_components):
            covs[k] = self._cov_process_(covs[k])
            denorm = np.max([8e-3, mvnormcdf(self.uppers, mus[k], covs[k], self.lowers)])
            normalizors.append(denorm)

        for i in range(N):
            ps_pos = [np.max([1e-10, mvnormcdf(pos_right[i, :], mus[k], covs[k], pos_left[i, :])])
                      for k in range(n_components)]
            prob = np.sum(ps_pos * comp_ws)
            if prob > 0:
                res += weights[i] * np.log(prob)
            else:
                res += -1.0 * np.NaN

        if debug_info is not None:
            debug_info.append(-res)
            if len(debug_info) % 200  == 0:
                print(len(debug_info), -res)

        return -1 * res

    def fit_mixture(self, pos_left, pos_right, weights, n_components=2, tol=1e-4, maxiter=4000, verbose=False):
        left, right = np.asarray(pos_left), np.asarray(pos_right)
        weights = np.asarray(weights)
        debugs = list() if verbose else None
        centers = (left + right) / 2.0
        init_gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                                   n_components=n_components, X=centers, weights=weights,
                                                   stop_threshold=0.01, n_jobs=2)

        init_mus, init_covs = list(), list()
        init_comp_ws = np.array(init_gmm.weights)
        init_comp_ws /= np.sum(init_comp_ws)
        for i in range(n_components):
            paras = init_gmm.distributions[i].parameters
            init_mus.append(np.array(paras[0]))
            init_covs.append(np.array(paras[1]))

        init_paras = self._paras_compose_(init_mus, init_covs, list(init_comp_ws))

        method = 'Nelder-Mead'
        res = opt.minimize(self._mixture_optpara, init_paras, args=(left, right, weights, n_components, debugs),
                           method = method, tol = tol, options={'maxiter': maxiter, 'disp': verbose})
        if verbose:
            print("Method:{}; Initial parameter: {};".format(method, init_paras))
            print("Converged Parameter: {}".format(res.x))

        mus, covs, comp_ws = self._paras_decompose_(res.x, n_components)
        return mus, covs, comp_ws, res.fun

    def log_loss(self, paras, pos_left, pos_right, weights, is_mix, n_components=2):
        loss = 0.0
        paras_vec = self._paras_compose_(paras['mus'], paras['covs'], paras['weights'])
        if is_mix:
            loss = self._mixture_optpara(paras_vec, pos_left, pos_right, weights, n_components)
        else:
            loss = self._single_optpara(paras_vec, pos_left, pos_right, weights)

        return loss
