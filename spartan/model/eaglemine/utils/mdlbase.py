#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  Some basic mdl (model description length) tools
#  Author: wenchieh
#
#  Project: eaglemine
#      mdlbase.py
#      Version:  1.0
#      Date: December 10 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/10/2017>
# 

__author__ = 'wenchieh'

# third-party lib
import numpy as np

class MDLBase(object):
    FLOATBIT = 4 * 8     # # reference to: KDD'14  AutoPlait

    @staticmethod
    def float_mdl(val, schema=1, precision=10):
        # 1. using the fixed code-length to encoding a float
        # 2. proportional to the magnitude of the value and the given precision \eta
        code_length = 0.0
        if schema == 1:
           code_length = MDLBase.FLOATBIT
        elif schema == 2:
            sign_code = 1
            val = abs(val)
            val_code = np.log2(val * 10**(precision)) if val > 0 else precision
            code_length = sign_code + val_code

        return code_length

    @staticmethod
    def integer_mdl(z):
        """
        encoded size of an integer >=1 as by Peter Elias's 1975 Elias code for integers
        Elias code variant for encoding zero.
        :param z: non-negative integer
        :return: encode length
        """
        if z < 0:
            ValueError("The Elias encoding can't cope with the negative integer. "
                       "Input non-negative integer value.")

        z += 1   #to encode 0, here we define the code-schema all value add 1
        c = 0
        while z > 1:
            t = np.floor(np.log2(z))
            c += 1 + int(t)
            z = t
        c += 1    # here 1 is the halt bit for encoding.
        return c

    @staticmethod
    def integer_elias_decode(code):
        if len(code) <= 0:
            ValueError("Code input error!")

        z = 0
        if len(code) == 1:
            z = 1
        else:
            code = code[:-1]
            sig_bits = 2
            while len(code) > 0:
                k = int(code[:sig_bits], 2)
                if sig_bits >= len(code):
                    z = k
                    break
                else:
                    code = code[sig_bits:]
                    sig_bits = k + 1
                    continue
        z -= 1   # encode all non-negative number.
        return z

    @staticmethod
    def integer_elias_encode(z):
        if z < 0:
            ValueError("The Elias encoding can't cope with the negative integer. "
                       "Input non-negative integer value.")
        z += 1
        code = ''
        val_code = bin(z)[2:]
        while len(val_code) > 1:
            code = val_code + code
            pre_len = len(val_code)
            val_code = bin(pre_len - 1)[2:]
        code += '0'    # the halt bit.
        return code

    @staticmethod
    def seq_diff_mdl(p, q):
        """
        encode the difference between two integer sequences.
        :param p: [int, array] the actual values
        :param q: [int, array] the expect values
        :return:
        """
        p, q = np.round(p), np.round(q)

        ## deprecated
        ## approximate Elias code (not the shortest code length)
        # code_len = np.sum([1 + 2 * np.ceil(np.log2(d)) if d > 0 else 1 for d in np.abs(p - q)])

        # absolute difference encode
        # code_len = np.sum(np.abs(p - q)) + len(p)

        ## here 1 is the sign bit, integer_mdl(d) is the code-length of absolute-difference.
        code_len = np.sum([1 + MDLBase.integer_mdl(d) for d in np.abs(p - q)])
        return code_len