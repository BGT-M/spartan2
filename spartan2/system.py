#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os
import sys
from abc import abstractmethod, ABCMeta
from .algorithm.graph import Holoscope, Eaglemine, Fraudar, SVDS
from .algorithm.time_series import Beatlex


class Engine:
    SINGLEMACHINE = "spartan2.models.SingleMachine"


class Model():
    __metaclass__ = ABCMeta

    def __init__(self):
        self.name = None
        self.tensorlist = None

    @abstractmethod
    def create(self, data, alg_obj, model_name):
        pass


class TraingleCount(Model):
    def create(self):
        pass


class AnomalyDetection(Model):
    def create(self, data: list, alg_obj: "function", model_name: str) -> "result of algorithm":
        alg_name = alg_obj.__name__
        alg_list = {
            'HOLOSCOPE': Holoscope,
            'EAGLEMINE': Eaglemine,
            'FRAUDAR': Fraudar,
        }
        return alg_list[alg_name](data, alg_obj, model_name)


class Decomposition(Model):
    def create(self, data: list, alg_obj: "function", model_name: str) -> "result of algorithm":
        alg_name = alg_obj.__name__
        alg_list = {
            'SVDS': SVDS,
        }
        return alg_list[alg_name](data, alg_obj, model_name)


class Timeseries(Model):
    def create(self, data: list, alg_obj: "function", model_name: str) -> "result of algorithm":
        alg_name = alg_obj.__name__
        alg_list = {
            'BEATLEX': Beatlex,
        }
        return alg_list[alg_name](data, alg_obj, model_name)
