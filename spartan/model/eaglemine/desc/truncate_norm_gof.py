#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  The Goodness-of-fit (GoF) for truncated normal distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      truncate_norm_gof.py
#      Version:  1.0
#      Date: December 21 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/21/2017>
#

__author__ = 'wenchieh'


# sys
import errno
import multiprocessing
from collections import Counter
from multiprocessing import Process
from multiprocessing.queues import Queue

# third-party lib
import numpy as np
from scipy.stats import norm, uniform
import scipy.optimize as opt


def _ad_alters_check(alternative):
    valid_alters = ["twosided", "less", "greater"]
    if alternative not in valid_alters:
        raise ValueError("Error: Invalid argument for alternative hypothesis, "
                   "which must be on of {}".format(valid_alters))

def _ad_stat_(n, zH, zs, alternative='twosided'):
    '''
    Supremum Class Anderson-Darling test
           Supremum class version of the Anderson-Darling test providing a comparison of a
         fitted distribution with the empirical distribution.
    :param n: the size of samples  #pts
    :param zH: the probability of point at truncate-clip point H, i.e. zH = \head F_{theta} (H)
    :param zs: the probability of each samples i.e. zs_j = \head F_{theta} (x_j)
               |zs| = n
    :param alternative: indicates the alternative hypothesis and must be one of
                    "twosided" (default), "less", or "greater".
                    Initial letter must be specified only.
    :return: supremum AD statistic
    '''
    assert(len(zs) == n)
    _ad_alters_check(alternative)

    AD = 0.0
    if alternative == 'greater':
        part = np.array([(zH + 1.0*(j+1)*(1-zH)/n - zs[j])/(np.sqrt((zs[j]-zH)*(1-zs[j]))) for j in range(n)])
        AD = np.sqrt(n) * np.max(part)
    elif alternative == 'less':
        part = np.array([(zs[j] - zH - 1.0*(j-1)*(1-zH)/n)/(np.sqrt((zs[j]-zH)*(1-zs[j]))) for j in range(n)])
        AD = np.sqrt(n) * np.max(part)
    else:
        part_p = np.array([(zH + 1.0*j*(1-zH)/n - zs[j])/(np.sqrt((zs[j]-zH)*(1-zs[j]))) for j in range(n)])
        part_m = np.array([(zs[j] - zH - 1.0*(j-1)*(1-zH)/n)/(np.sqrt((zs[j]-zH)*(1-zs[j]))) for j in range(n)])
        AD = np.sqrt(n) * np.max([part_p, part_m])

    return AD

def _aduptail_stat_(n, zH, zs, alternative='twosided'):
    '''
    Supremum Class Upper Tail Anderson-Darling test
           Supremum class version of the Upper Tail Anderson-Darling test providing a comparison
         of a fitted distribution with the empirical distribution.
    :param n: the size of samples  #pts
    :param zH: the probability of point at truncate-clip point H, i.e. zH = \head F_{theta} (H)
    :param zs: the probability of each samples i.e. zs_j = \head F_{theta} (x_j)
               |zs| = n
    :param alternative: indicates the alternative hypothesis and must be one of
                    "twosided" (default), "less", or "greater".
                    Initial letter must be specified only.
    :return: supremum AD statistic
    '''
    assert(len(zs) == n)
    _ad_alters_check(alternative)

    ADUP = 0.0
    if alternative == 'greater':
        part = np.array([(zH + 1.0*(1-zH)*(j+1)/n - zs[j])/(1-zs[j]) for j in range(n)])
        ADUP = np.sqrt(n) * np.max(part)
    elif alternative == 'less':
        part = np.array([(zs[j] - zH - 1.0*j*(1-zH)/n)/(1-zs[j]) for j in range(n)])
        ADUP = np.sqrt(n) * np.max(part)
    else:
        part_p = np.array([(zH + 1.0*(1-zH)*(j+1)/n - zs[j])/(1-zs[j]) for j in range(n)])
        part_m = np.array([(zs[j] - zH - 1.0*j*(1-zH)/n)/(1-zs[j]) for j in range(n)])
        ADUP = np.sqrt(n) * np.max([part_p, part_m])

    return ADUP

def _ad2_stat_(n, zH, zs):
    '''
    the Quadratic Class Anderson-Darling Test Statistic for left-truncated data
           Quadratic class Anderson-Darling test providing a comparison of a fitted
         distribution with the empirical distribution.
    :param n:  the size of samples  #pts
    :param zH: the probability of point at truncate-clip point H, i.e. zH = \head F_{theta} (H)
    :param zs: the probability of each samples i.e. zs_j = \head F_{theta} (x_j)
               |zs| = n
    :return: AD2 statistic
    '''
    assert(len(zs) == n)
    part = np.sum([(1+2*(n-(j+1)))*np.log(1-zs[j]) - (1-2*(j + 1))*np.log(zs[j] - zH) for j in range(n)])
    AD2 = -n + 2*n*np.log(1-zH) - part * 1.0 / n
    return AD2

def _ad2uptail_stat_(n, zH, zs):
    '''
    the Quadratic Class Upper Tail Anderson-Darling test
           Quadratic Class Upper Tail Anderson-Darling test providing a comparison of a fitted
        distribution with the empirical distribution.
    :param n:  the size of samples  #pts
    :param zH: the probability of point at truncate-clip point H, i.e. zH = \head F_{theta} (H)
    :param zs: the probability of each samples i.e. zs_j = \head F_{theta} (x_j)
               |zs| = n
    :return: AD2UPTAIL statistic
    '''
    assert(len(zs) == n)
    part = np.sum([(1 + 2*(n-(j+1)))/(1 - zs[j]) for j in range(n)])
    AD2UP = -2*n*np.log(1-zH) + 2*np.sum(np.log(1 - zs)) + (1.0 - zH) / n * part
    return AD2UP

def retry_on_eintr(function, *args, **kw):
    while True:
        try:
            return function(*args, **kw)
        except IOError as e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise


class RetryQueue(Queue):
    def __init__(self):
        super(RetryQueue, self).__init__(ctx=multiprocessing.get_context())

    def get(self, block=True, timeout=None):
        return retry_on_eintr(Queue.get, self, block, timeout)


class TruncatedNormalGoF(object):
    '''
    Reference:
        Chernobay, A., Rachev, S., Fabozzi, F. (2005), Composites goodness-of-fit tests
            for left-truncated loss samples, Tech. rep., University of Calivornia Santa Barbara
    '''

    def __init__(self, n_jobs=1):
        self._n_jobs_ = n_jobs

    def _optimize_mle_(self, init_pram, xs, weights, H, epsilon=1e-10):
        n = len(xs)
        mean, std =  init_pram[0], init_pram[1]

        dens = 1 - norm.cdf(H, loc=mean, scale=std) + epsilon
        numerator = norm.pdf(xs, loc=mean, scale=std)
        xs_pdf = numerator / dens
        wtxs_pdf = weights * xs_pdf
        val = np.asarray(wtxs_pdf[xs > H])
        if val.all() > 0:
            res = -1.0 * np.sum(np.log(val))
        else:
            res = -1.0 * np.NaN

        if np.isinf(res):
            res = - np.log(np.finfo('float64').resolution) * n

        return res

    def _sample_fit_(self, xs, init_mean, init_std, H, weights=None,
                     tol=1e-4, maxiter=4000, verbose=False):
        n = len(xs)

        if weights is None:
            weights = np.array([1] * n)
        init_parm = np.array([init_mean, init_std])
        res = opt.minimize(self._optimize_mle_, init_parm, args=(xs, weights, H),
                           method = 'BFGS', tol = tol,
                           options={'maxiter': maxiter, 'disp': verbose})

        mean, std = res.x[0], res.x[1]
        return mean, std


    def _mc_simulate_(self, n, mean, std, zH, H, STATSITIC_func, sims,
                      multi=False, queue=0, jobno=0):
        np.random.seed(jobno)
        sz = list()
        # 1. generate large number of samples (sim) from the fitted truncated distribution of size n
        for i in range(sims):
            uniszs = uniform.rvs(size=n)
            sz_k = norm.ppf(uniszs*(1 - zH) + zH, loc=mean, scale=std)
            sz.append(sz_k)
        # 2. Fit truncated distribution and estimate conditional parameters $\head \theta$
        mc_TS = list()
        for i in range(sims):
            mc_mean, mc_std = self._sample_fit_(sz[i], mean, std, H)

        # 3. Estimate the GOF statistic value Di for each sample
            mc_zH = norm.cdf(H, loc=mc_mean, scale=mc_std)

            uni_szi = sorted(np.unique(sz[i]))
            count = Counter(sz[i])
            mc_zs = list()
            for mczj in uni_szi:
                mczj_cdf = norm.cdf(mczj, loc=mc_mean, scale=mc_std)
                mc_zs.extend([mczj_cdf] * count[mczj])
            mc_zs = np.asarray(mc_zs)
            # mc_zs = np.array([norm.cdf(mczj, loc=mc_mean, scale=mc_std) for mczj in sorted(sz[i])])
            mc_TS.append(STATSITIC_func(len(sz[i]),mc_zH, mc_zs))

        if multi:
            queue.put(mc_TS)
        else:
            return mc_TS

    def _multimc_simus_(self, n, mean, std, zH, H, STATSITIC_func, sims):
        res = list()
        queues = [RetryQueue() for i in range(self._n_jobs_)]
        jobs_sims = int(np.ceil(sims * 1.0 / self._n_jobs_))
        args = [(n, mean, std, zH, H, STATSITIC_func, jobs_sims, True, queues[i], i + 1)
                for i in range(self._n_jobs_)]
        jobs = [Process(target=self._mc_simulate_, args=(a)) for a in args]
        for j in jobs: j.start()
        for q in queues: res.append(q.get())
        for j in jobs:  j.join()

        return np.hstack(res)


    def _montecarlo_test_(self, xs, mean, std, H, STATSITIC_func, sims, tol):
        '''
        Monte-Carlo simulation based GoF test
               Performs Monte-Carlo based Goodness-of-Fit tests.
               the mctest is called by the GoF tests defined in this package. For internal use only.
        :return:
        '''
        if min(xs) < H:
            raise ValueError("'min(xs)' must be greater or equal to 'H'")

        p_value = 0.0
        n = len(xs)
        zH = norm.cdf(H, loc=mean, scale=std)
        uni_xs = sorted(np.unique(xs))
        count = Counter(xs)
        zs = list()
        for zj in uni_xs:
            zjcdf = norm.cdf(zj, loc=mean, scale=std)
            zs.extend([zjcdf] * count[zj])
        zs = np.asarray(zs)
        # zs = np.array([norm.cdf(zj, loc=mean, scale=std) for zj in sorted(xs)])
        TS0 = STATSITIC_func(n, zH, zs)

        if not np.isfinite(TS0):  #(np.abs(TS0) == np.inf) or (TS0 is np.NaN):
            # raise ValueError("test statistic value can't be calculated")
            print("Warning: test statistic value can't be calculated")
            return p_value, TS0, 0

        # # 1. generate large number of samples (sims) from the fitted truncated distribution of size n
        # np.random.seed(1024)
        # sz = list()
        # for i in range(sims):
        #     uniszs = uniform.rvs(size=n)
        #     sz_k = norm.ppf(uniszs*(1 - zH) + zH, loc=mean, scale=std)
        #     sz.append(sz_k)
        # # 2. Fit truncated distribution and estimate conditional parameters $\head \theta$
        # mc_TS = list()
        # for i in range(sims):
        #     mc_mean, mc_std = self._sample_fit_(sz[i], mean, std, H)
        #
        # # 3. Estimate the GOF statistic value Di for each sample
        #     mc_zH = norm.cdf(H, loc=mc_mean, scale=mc_std)
        #     mc_zs = np.array([norm.cdf(mczj, loc=mc_mean, scale=mc_std) for mczj in sorted(sz[i])])
        #     mc_TS.append(STATSITIC_func(len(sz[i]),mc_zH, mc_zs))

        mc_TS = list()
        # 1. generate large number of samples (sims) from the fitted truncated distribution of size n
        # 2. Fit truncated distribution and estimate conditional parameters $\head \theta$
        # 3. Estimate the GOF statistic value Di for each sample
        if self._n_jobs_ > 1:
            mc_TS = self._multimc_simus_(n, mean, std, zH, H, STATSITIC_func, sims)
        else:
            mc_TS = self._mc_simulate_(n, mean, std, zH, H, STATSITIC_func, sims)

        # 4. Calculate p-value as the proportion of times the sample statistic values
        #     exceed the observed value D of the original samples
        k, z0, z1 = 0, 0, 2
        suc_nsims = 0
        for i in range(len(mc_TS)):
            if mc_TS[i] > TS0:
                k += 1
            z0 = z1
            z1 = k * 1.0 / (i + 1)
            suc_nsims += 1
            if (np.abs(z1 - z0) < tol) and (np.abs(z1 - 0.5) < 0.5):
                break
        p_value = z1 + 1e-20

        # 5. Reject H0 if the p-value is smaller than \alpha
        return p_value, TS0, suc_nsims


    def ad_test(self, xs, mean, std, H=np.NaN, alternative='twosided', sim=100, tol=1e-4):
        _ad_alters_check(alternative)
        if H is np.NAN:
            H = -np.inf

        def ad_stat(n, zH, zs):
            return _ad_stat_(n, zH, zs, alternative)

        p_value, TS, nsims = self._montecarlo_test_(xs, mean, std, H, ad_stat, sim, tol)

        ans = {'method': 'Supremum Class Anderson-Darling Test',
               'statitic_name': 'AD', 'statistic': TS, 'alternative': alternative,
               'p_value': p_value, 'nsims': nsims, 'threshold': H}

        return ans

    def adup_test(self, xs, mean, std, H=np.NaN, alternative='twosided',
                sim=100, tol=1e-4):
        _ad_alters_check(alternative)
        if H is np.NAN:
            H = -np.inf

        def aduptail_stat(n, zH, zs):
            return _aduptail_stat_(n, zH, zs, alternative)

        p_value, TS, nsims = self._montecarlo_test_(xs, mean, std, H, aduptail_stat, sim, tol)

        stat_name = { 'twosided': 'ADup', 'less': 'ADup-', 'greater': 'ADup+'  }
        ans = {'method': 'Anderson-Darling Upper Tail Test',
               'statistic_name': stat_name[alternative], 'statistic': TS, 'alternative': alternative,
               'p_value': p_value, 'nsims': nsims, 'threshold': H}
        return ans

    def ad2_test(self, xs, mean, std, H=np.NaN, sim=100, tol=1e-4):
        if H is np.NAN:
            H = -np.inf
        p_value, TS, nsims = self._montecarlo_test_(xs, mean, std, H, _ad2_stat_, sim, tol)
        ans = {'method':  'Quadratic Class Anderson-Darling Test',
               'statistic_name': 'AD2', 'statistic': TS,
               'p_value': p_value, 'nsims': nsims, 'threshold': H}
        return ans

    def ad2up_test(self, xs, mean, std, H=np.NaN, sim=100, tol=1e-4):
        if H is np.NAN:
            H = -np.inf
        p_value, TS, nsims = self._montecarlo_test_(xs, mean, std, H, _ad2uptail_stat_, sim, tol)
        ans = {'method':  'Quadratic Class Anderson-Darling Test',
               'statistic_name': 'AD2up', 'statistic': TS,
               'p_value': p_value, 'nsims': nsims, 'threshold': H}
        return ans
