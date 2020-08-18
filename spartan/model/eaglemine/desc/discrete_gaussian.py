#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Describe hypercubes (or only Two-dimensional) with discrete Gaussian distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      discrete_gaussian.py
#      Version:  1.0
#      Date: November 17 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/17/2017>
# 
__author__ = 'wenchieh'

# third-party lib
import numpy as np
from pomegranate import *
from scipy.stats import multivariate_normal


##################################################################################
#
#  A tutorial for package pomegranate is reference to:
#     [1] pomegranate: probabilistic modelling in python 
#         https://homes.cs.washington.edu/~jmschr/lectures/pomegranate.html
#
##################################################################################

class DiscreteGaussian(object):
    def fit_single(self, pos_left, pos_right, weights):
        left, right = np.asarray(pos_left), np.asarray(pos_right)
        centers = (left + right) / 2.0
        fit_model = MultivariateGaussianDistribution.from_samples(centers, weights=weights)
        mu, cov = np.array(fit_model.parameters[0]), np.array(fit_model.parameters[1])

        _pdfs_ = multivariate_normal.logpdf(centers, mu, cov, allow_singular=True)
        loss_log = np.sum([weights[i] * _pdfs_[i] for i in range(len(centers))])
        return [mu], [cov], [1.0], loss_log

    def fit_mixture(self, pos_left, pos_right, weights, n_components=2):
        left, right = np.asarray(pos_left), np.asarray(pos_right)
        centers = (left + right) / 2.0
        fit_gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
                                                   n_components=n_components, X=centers, weights=weights,
                                                   stop_threshold=0.01, n_jobs=2)
        mus, covs = list(), list()
        comp_ws = np.array(fit_gmm.weights)
        comp_ws /= np.sum(comp_ws)
        for i in range(n_components):
            paras = fit_gmm.distributions[i].parameters
            mus.append(np.array(paras[0]))
            covs.append(np.array(paras[1]))

        _pdfs_ = fit_gmm.log_probability(centers)
        loss_log = np.sum([weights[i] * _pdfs_[i] for i in range(len(centers))])
        return mus, covs, comp_ws, loss_log


    def log_loss(self, paras, pos_left, pos_right, weights, is_mix, n_components=2):
        left, right = np.asarray(pos_left), np.asarray(pos_right)
        if is_mix is True:
            mus, covs, comp_ws = paras['mus'], paras['covs'], paras['weights']
            n_mus = len(mus)
            if n_mus != n_components:
                print("Warning: input parameter not consistent for mixture Gaussians.")
                print("\t n_component {}, #mu {}s.".format(n_components, len(mus)))
                if n_mus < n_components:
                    print("Error: n_component < #mu (#cov, #weihght), and exit.")
                    exit(0)

            comp_dists = list()
            for i in range(n_components):
                idist = MultivariateGaussianDistribution(mus[i], covs[i])
                comp_dists.append(idist)

            gmm = GeneralMixtureModel(comp_dists, weights=np.array(comp_ws))

            centers = (left + right) / 2.0
            pdfs = gmm.log_probability(centers)
            # loss_log = np.sum([weights[i] * pdfs[i] for i in range(len(centers))])
        else:
            mu, cov = paras['mus'][0], paras['covs'][0]
            centers = (np.array(pos_left) + np.array(pos_right)) / 2.0
            pdfs = multivariate_normal.logpdf(centers, mu, cov, allow_singular=True)

        loss_log = -1.0 * np.sum([weights[i] * pdfs[i] for i in range(len(centers))])
        return loss_log
