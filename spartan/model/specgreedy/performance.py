#!/usr/bin/env python3
# -*- coding=utf-8 -*-


def jaccard(pred, actual):
    intersectSize = len(set.intersection(set(pred[0]), set(actual[0]))) + \
                    len(set.intersection(set(pred[1]), set(actual[1])))
    unionSize = len(set.union(set(pred[0]), set(actual[0]))) + \
                len(set.union(set(pred[1]), set(actual[1])))
    return intersectSize *1.0 / unionSize

def getPrecision(pred, actual):
    intersectSize = len(set.intersection(set(pred[0]), set(actual[0]))) + \
                    len(set.intersection(set(pred[1]), set(actual[1])))
    return intersectSize *1.0 / (len(pred[0]) + len(pred[1]))

def getRecall(pred, actual):
    intersectSize = len(set.intersection(set(pred[0]), set(actual[0]))) + \
                    len(set.intersection(set(pred[1]), set(actual[1])))
    return intersectSize *1.0 / (len(actual[0]) + len(actual[1]))

def getFMeasure(pred, actual):
    prec = getPrecision(pred, actual)
    rec = getRecall(pred, actual)
    return 0 if (prec + rec == 0) else (2.0 * prec * rec / (prec + rec))

def getDimPrecision(pred, actual, idx):
    intersectSize = len(set.intersection(set(pred[idx]), set(actual[idx])))
    return intersectSize * 1.0 / len(pred[idx])

def getDimRecall(pred, actual, idx):
    intersectSize = len(set.intersection(set(pred[idx]), set(actual[idx])))
    return intersectSize * 1.0 / len(actual[idx])

def getDimFMeasure(pred, actual, idx):
    prec = getRowPrecision(pred, actual, idx)
    rec = getRowRecall(pred, actual, idx)
    return 0 if (prec + rec == 0) else (2.0 * prec * rec / (prec + rec))
