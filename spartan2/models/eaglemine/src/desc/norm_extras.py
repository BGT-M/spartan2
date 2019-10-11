#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#      norm_extras.py
#           some useful sub-routines for Aderson-Darling Test, multi-variate normal CDF
#      Version:  1.0
#      Goal: Routine scripts
#      Created by @wenchieh  on <11/30/2017>


__author__ = 'wenchieh'

# third-party lib
import numpy as np
from scipy.stats import mvn
from scipy.stats import distributions
from collections import namedtuple
from statsmodels.stats.weightstats import DescrStatsW

# Values from Stephens, M A, "EDF Statistics for Goodness of Fit and
#             Some Comparisons", Journal of the American Statistical
#             Association, Vol. 69, Issue 347, Sept. 1974, pp 730-737
_Avals_norm = np.array([0.576, 0.656, 0.787, 0.918, 1.092])

AndersonResult = namedtuple('AndersonResult', ('statistic',
                                               'critical_values',
                                               'significance_level'))

INFORMCODE = {0: 'normal completion with ERROR < EPS',
              1: '''completion with ERROR > EPS and MAXPTS function values used;
                    increase MAXPTS to decrease ERROR;''',
              2: 'N > 500 or N < 1'}


def norm_anderson(x, weights=None):
    """
    Anderson-Darling test for data coming from a particular distribution

    The Anderson-Darling test is a modification of the Kolmogorov-
    Smirnov test `kstest` for the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.
    For the Anderson-Darling test, the critical values depend on
    which distribution is being tested against.  This function works
    for normal, exponential distributions.

    Parameters
    ----------
    x : array_like
        array of sample data
    weight: array_like optional (default = None)
        array of weight for each sample data

    Returns
    -------
    statistic : float
        The Anderson-Darling test statistic
    critical_values : list
        The critical values for this distribution
    significance_level : list
        The significance levels for the corresponding critical values
        in percents.  The function returns critical values for a
        differing set of significance levels depending on the
        distribution that is being tested against.

    Notes
    -----
    Critical values provided are for the following significance levels:

    normal/exponential
        15%, 10%, 5%, 2.5%, 1%

    If A2 is larger than these critical values then for the corresponding
    significance level, the null hypothesis that the data come from the
    chosen distribution can be rejected.

    References
    ----------
    .. [1] http://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and
           Some Comparisons, Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [3] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
           Statistics with Unknown Parameters, Annals of Statistics, Vol. 4,
           pp. 357-369.
    .. [4] Stephens, M. A. (1977). Goodness of Fit for the Extreme Value
           Distribution, Biometrika, Vol. 64, pp. 583-588.
    .. [5] Stephens, M. A. (1977). Goodness of Fit with Special Reference
           to Tests for Exponentiality , Technical Report No. 262,
           Department of Statistics, Stanford University, Stanford, CA.
    .. [6] Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution
           Based on the Empirical Distribution Function, Biometrika, Vol. 66,
           pp. 591-595.
    """
    N = len(x)
    x = np.asarray(x)
    if weights is None:
        weights = np.ones(N)
    else:
        if len(weights) != N:
            raise ValueError("Invalid weights; the weights must "
                             "have the same shape as x or None.")
        weights = np.asarray(weights)

    sorted_index = np.argsort(x)
    x, weights = x[sorted_index], weights[sorted_index]

    xw = DescrStatsW(x, weights=weights, ddof=1)
    xbar = xw.mean
    s = xw.std

    w = (x - xbar) / s
    logcdf = distributions.norm.logcdf(w)
    logsf = distributions.norm.logsf(w)

    expand_lgcdf, expand_lgsf = [], []
    x_expand = []
    for i in range(len(x)):
        expand_lgcdf.extend([logcdf[i]]*weights[i])
        expand_lgsf.extend([logsf[i]]*weights[i])
        x_expand.extend([x[i]] * weights[i])

    print(len(x_expand))
    expand_lgcdf, expand_lgsf = np.array(expand_lgcdf), np.array(expand_lgsf)
    N_w = int(np.sum(weights))
    i = np.arange(1, N_w + 1)

    A2 = -N_w - np.sum((2*i - 1.0) / N_w * (expand_lgcdf + expand_lgsf[::-1]), axis=0)
    sig = np.array([15, 10, 5, 2.5, 1])
    critical = np.around(_Avals_norm / (1.0 + 4.0/N_w - 25.0/N_w/N_w), 3)

    return AndersonResult(A2, critical, sig)


def _mvstdnormcdf(lower, upper, corrcoef, **kwds):
    """
    standardized multivariate normal cumulative distribution function
    This is a wrapper for scipy.stats.kde.mvn.mvndst which calculates
    a rectangular integral over a standardized multivariate normal
    distribution.

    This function assumes standardized scale, that is the variance in each dimension
    is one, but correlation can be arbitrary, covariance = correlation matrix
    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    corrcoef : float or array_like
       specifies correlation matrix in one of three ways, see notes
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.
    Returns
    -------
    cdfvalue : float
        value of the integral
    Notes
    -----
    The correlation matrix corrcoef can be given in 3 different ways
    If the multivariate normal is two-dimensional than only the
    correlation coefficient needs to be provided.
    For general dimension the correlation matrix can be provided either
    as a one-dimensional array of the upper triangular correlation
    coefficients stacked by rows, or as full square correlation matrix
    See Also
    --------
    mvnormcdf : cdf of multivariate normal distribution without
        standardization
    :return:
    """
    n = len(lower)
    lower = np.array(lower)
    upper = np.array(upper)
    correl = np.zeros(n * (n - 1) // 2)
    corrcoef = np.array(corrcoef)

    if (lower.ndim != 1) or (upper.ndim != 1):
        raise ValueError("Error: can handle only 1D bounds")
    if len(upper) != n:
        raise ValueError("Error: bounds have different lengths")
    if n == 2 and corrcoef.size == 1:
        correl = corrcoef
        # print('case scalar rho', n)
    elif corrcoef.ndim == 1 and len(corrcoef) == n * (n - 1) / 2.0:
        # print('case flat corr', corrcoeff.shape)
        correl = corrcoef
    elif corrcoef.shape == (n, n):
        # print('case square corr',  correl.shape)
        correl = corrcoef[np.tril_indices(n, -1)]
    else:
        raise ValueError("Error: corrcoef has incorrect dimension")

    if not 'maxpts' in kwds:
        if n > 2:
            kwds['maxpts'] = 10000 * n

    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0 * np.ones(n)

    np.putmask(infin, lowinf, 0)  # infin.putmask(0,lowinf)
    np.putmask(infin, uppinf, 1)  # infin.putmask(1,uppinf)
    # this has to be last
    np.putmask(infin, lowinf * uppinf, -1)

    ##    #remove infs
    ##    np.putmask(lower,lowinf,-100)# infin.putmask(0,lowinf)
    ##    np.putmask(upper,uppinf,100) #infin.putmask(1,uppinf)

    error, cdfvalue, inform = mvn.mvndst(lower, upper, infin, correl, **kwds)
    if inform:
        print("Error Something wrong. {}: {}".format(INFORMCODE[inform], error))
    return cdfvalue


def mvnormcdf(upper, mu, cov, lower=None, **kwds):
    '''multivariate normal cumulative distribution function
    This is a wrapper for scipy.stats.kde.mvn.mvndst which calculates
    a rectangular integral over a multivariate normal distribution.

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    mu : array_lik, 1d
       list or array of means
    cov : array_like, 2d
       specifies covariance matrix
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.
    Returns
    -------
    cdfvalue : float
        value of the integral
    Notes
    -----
    This function normalizes the location and scale of the multivariate
    normal distribution and then uses `mvstdnormcdf` to call the integration.
    See Also
    --------
    mvstdnormcdf : location and scale standardized multivariate normal cdf
    '''
    upper = np.array(upper)
    if lower is None:
        lower = -np.ones(upper.shape) * np.inf
    else:
        lower = np.array(lower)
    cov = np.array(cov)
    stdev = np.sqrt(np.diag(cov))
    lower = (lower - mu) / stdev
    upper = (upper - mu) / stdev
    divrow = np.atleast_2d(stdev)
    corr = cov / divrow / divrow.T
    # v/np.sqrt(np.atleast_2d(np.diag(covv)))/np.sqrt(np.atleast_2d(np.diag(covv))).T

    return _mvstdnormcdf(lower, upper, corr, **kwds)
