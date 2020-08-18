#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Describe cells with Discrete Truncate Multivariate Normal(Gaussian) distribution
#  Author: wenchieh
#
#  Project: eaglemine
#      dtmnorm_describe.py
#      Version:  1.0
#      Date: December 1 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/1/2017>


__author__ = 'wenchieh'

# third-party lib
import numpy as np

# project
from .dtmnorm import DTMNorm


class DTMNormDescribe(object):
    _SERIALIZE_DELIMITER_ = ';'
    ZERO_BOUND = 0
    INF_BOUND = np.inf

    def __init__(self, ndim=2, is_mix=False, n_components=2):
        self.is_mix = is_mix
        self.ndim = ndim
        self.paras = None
        self._npos = None
        self.data = None
        self.values = None
        self._total_values = None
        self.n_components = max([n_components, 2]) if is_mix else 1

    def set_bounds(self, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = np.array([self.ZERO_BOUND] * self.ndim)
        if upper_bound is None:
            upper_bound = np.array([self.INF_BOUND] * self.ndim)
        self.descriptor = DTMNorm(self.ndim, lower_bound, upper_bound)

    def set_data(self, cells, values):
        self.data = np.asarray(cells, float)
        self.values = np.asarray(values)
        # self.values = np.asarray(values, int)
        self._npos, self._total_values = len(self.data), np.sum(self.values)

    def get_bounds(self):
        return self.descriptor.get_lowers(), self.descriptor.get_uppers()

    def fit(self, cells, values, type='left'):
        """
        fit the hypercubes with edge-length = 1 with Truncated Gaussian descriptor
        :param cells: input cells,  which dimension >= 2
        :param values: #point in each cell
        :param type: indicate the cells position,  "center" | "left" | "right"
                     The coordinate given in cell is at ["center" | "left" | "right"] of affiliated cell
        :return:
        """
        n = np.ndim(cells)
        if n != self.ndim:
            RuntimeWarning("Warning: Input data dimension iss not consistence with initialization!")
            if n > self.ndim:
                raise ValueError("Error: input data dimension ({}) exceed initial dimension ({})".format(n, self.ndim))

        self.set_data(cells, values)

        left, right = self.data, self.data + 1
        if type == 'center':
            left, right = self.data - 0.5, self.data + 0.5
        elif type == 'right':
            left, right = self.data - 1, self.data
        else:
            pass

        if self.is_mix:
            mus, covs, ws, loss = self.descriptor.fit_mixture(left, right, self.values, self.n_components)
        else:
            mus, covs, ws, loss = self.descriptor.fit_single(left, right, self.values)

        self.paras = {'mus': mus, 'covs':covs, 'weights':ws, 'loss':loss}

    def apply(self, cells, values, type='left', loss_update=False):
        # values = np.asarray(values, int)
        values = np.asarray(values)
        if self.paras is not None:
            pos = np.asarray(cells)
            m, n = pos.shape
            if n != self.ndim:
                RuntimeWarning("Warning: Input data dimension iss not consistence with initialization!")
                if n > self.ndim:
                    raise ValueError("Error: input data dimension ({}) exceed initial dimension ({})".format(n, self.ndim))
            else:
                left, right = pos, pos + 1
                if type == 'center':
                    left, right = pos - 0.5, pos + 0.5
                elif type == 'right':
                    left, right = pos - 1, self.data
                else:
                    pass
                loss = self.descriptor.log_loss(self.paras, left, right, values, self.is_mix, self.n_components)
                if loss_update:
                    self.paras["loss"] = loss
                return loss
        else:
            raise RuntimeError("No derived fitting parameters!")

    def dump(self):
        if self.paras is not None:
            print("descriptor information:")
            print("#dimension: {}, #pos: {}, total count: {}".format(self.ndim, self._npos, self._total_values))
            print("parameters: {}".format("mixture (#{})".format(self.n_components) if self.is_mix else "single"))
            print("\t mus:{}, covs: {}, weights: {}, loss: {}".format(self.paras["mus"], self.paras["covs"],
                                                                      self.paras["weights"], self.paras["loss"]))
        else:
            raise RuntimeError("No data fit")
        print("done!\n")

    def __str__(self):
        content = ""
        if self.paras is not None:
            mixture = 1 if self.is_mix is True else 0
            mus_lst, covs_lst = list(), list()
            for mu in self.paras["mus"]:
                mus_lst.append(np.asarray(mu).tolist())
            for cov in self.paras["covs"]:
                covs_lst.append(np.asarray(cov).tolist())
            ws_str = str(np.asarray(self.paras["weights"]).tolist())
            content += str(self.ndim) + self._SERIALIZE_DELIMITER_ + str(mixture) + \
                       self._SERIALIZE_DELIMITER_ + str(mus_lst) + self._SERIALIZE_DELIMITER_ + \
                       str(covs_lst) + self._SERIALIZE_DELIMITER_ + ws_str + self._SERIALIZE_DELIMITER_ + \
                       str(self.paras["loss"]) + self._SERIALIZE_DELIMITER_ + str(len(self.data)) + \
                       self._SERIALIZE_DELIMITER_ + str(np.sum(self.values))
        return content

    def load(self, desc_str, verbose=False):
        toks = desc_str.strip().split(self._SERIALIZE_DELIMITER_)
        self.ndim = int(toks[0])
        mixture = int(toks[1])
        self.is_mix = True if mixture == 1 else False
        self.paras = dict()
        self.paras["mus"] = np.asarray(eval(toks[2]))
        self.paras["covs"] = np.asarray(eval(toks[3]))
        self.paras["weights"] = np.asarray(eval(toks[4]))
        self.paras["loss"] = float(toks[5])
        self._npos = int(toks[6])
        self._total_values = float(toks[7]) #int(toks[7])
        self.n_components = len(self.paras["mus"])
        self.set_bounds()
        if verbose:
            self.dump()

    def compact_parm(self):
        parms = list()
        parms.extend(list(np.asarray(self.paras['mus']).squeeze().flatten()))
        if self.is_mix:
            parms.extend(list(self.paras['weights'][:-1]))
        for k in range(self.n_components):
            cov = self.paras['covs'][k]
            for m in range(self.ndim):
                parms.extend(cov[m, m:])
        return parms