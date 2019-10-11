#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Viki Zhao

import os, sys
from .algorithm import Holoscope, Eaglemine, Fraudar, SVDS

alg_list = {
    "AnomalyDetection":{
        "HOLOSCOPE": "HOLOSCOPE",
        "FRAUDAR": "FRAUDAR",
        "EAGLEMINE": "EAGLEMINE"
    },
    "EigenDecompose":{
        "SVDS": "SVDS"
    },
    "TriangleCount":{
        "THINKD": "THINKD"
    }
}

class Engine:
    SINGLEMACHINE = "spartan2.models.SingleMachine"

class Model():
    def __init__(self):
        self.name = None
        self.edgelist = None

class TraingleCount(Model):
    def create(self):
        pass

class AnomalyDetection(Model):
    def create(self, edgelist, alg_obj, model_name):
        alg_name = str(alg_obj)
        if alg_name.find(alg_list["AnomalyDetection"]["HOLOSCOPE"]) != -1:
            return Holoscope(edgelist, alg_obj, model_name)
        elif alg_name.find(alg_list["AnomalyDetection"]["EAGLEMINE"]) != -1:
            return Eaglemine(edgelist, alg_obj, model_name)
        elif alg_name.find(alg_list["AnomalyDetection"]["FRAUDAR"]) != -1:
            return Fraudar(edgelist, alg_obj, model_name)

class EigenDecompose(Model):
    def create(self, edgelist, alg_obj, model_name):
        alg_name = str(alg_obj)
        if alg_name.find(alg_list["EigenDecompose"]["SVDS"]) != -1:
            return SVDS(edgelist, alg_obj, model_name)

