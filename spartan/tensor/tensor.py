#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tensor.py
@Desc    :   Interface of class STTensor.
'''

# here put the import lib


class STTensor:
    pass


#TODO => __str__ of STTensor
def printTensorInfo(tensorlist, hasvalue):
    m = len(tensorlist[0]) - hasvalue
    print(f"Info: Tensor is loaded\n\
           ----------------------\n\
             attr     |\t{m}\n\
             values   |\t{hasvalue}\n\
             nonzeros |\t{len(tensorlist)}\n")
