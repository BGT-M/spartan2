#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Describe the milti-dimensional islands (hypercube centered) with gaussian model.
#  Author: wenchieh
#
#  Project: eaglemine
#      gaussian_describe.py
#      Version:  1.0
#      Date: December 12 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/12/2017>
#

__author__ = 'wenchieh'

# third-party lib
import numpy as np

# project
from .discrete_gaussian import DiscreteGaussian


class GaussianDescribe(object):
    _SERIALIZE_DELIMITER_ = ';'
    def __init__(self, ndim=2, is_mix=False, n_components=2):
        self.descriptor = DiscreteGaussian()
        self.is_mix = is_mix
        self.ndim = ndim
        self.paras = None
        self._npos = None
        self.data = None
        self.values = None
        self._total_values = None
        self.n_components = n_components if is_mix else 1

    def set_data(self, hypercubes, values):
        self.data = np.asarray(hypercubes, float)
        self.values = np.asarray(values)
        # self.values = np.asarray(values, int)
        self._npos, self._total_values = len(self.data), np.sum(self.values)

    def fit(self, hypercubes, values, type='left'):
        """
        fit the hypercubes with edge-length = 1 with Gaussian descriptor
        :param hypercubes: input hypercubes,  which dimension >= 2
        :param values: value in each hypercube (maybe #point or weight)
        :param type: indicate the hypercube position,  "center" | "left" | "right"
                     The coordinate given in hypercube is at ["center" | "left" | "right"] of affiliated hypercubes
        :return:
        """
        self.set_data(hypercubes, values)
        self.ndim = np.ndim(self.data)

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
        # print("data fit done!")

    def apply(self, hypercubes, values, type='left', loss_update=False):
        values = np.asarray(values, int)
        if self.paras is not None:
            pos = np.asarray(hypercubes)
            m, n = pos.shape
            if n != self.ndim:
                RuntimeWarning("Input data dimension iss not consistence with initialization!")
                if n > self.ndim:
                    raise ValueError("Input data dimension ({}) exceed initial dimension ({})".format(n, self.ndim))
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
            print("parameters: {}".format("mixture(#{})".format(self.n_components) if self.is_mix else "single"))
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
        self._total_values = float(toks[7])
        self.n_components = len(self.paras["mus"])
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