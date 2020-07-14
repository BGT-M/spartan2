#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_ioutil.py
@Desc    :   Test file for ioutil.
'''

# here put the import lib

import unittest

from . import loadTensor, TensorData


class TestLoadTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Start Test")

    @classmethod
    def tearDownClass(cls):
        print("End of Test")

    def test_load_tensor(self):
        data = loadTensor("./live-tutorials/inputData/example.tensor", col_types=[int, int])
        self.assertIsInstance(data, TensorData)

    def test_load_tensor2(self):
        data = loadTensor("./live-tutorials/inputData/example_graph.tensor", col_types=[int, int, int])
        self.assertIsInstance(data, TensorData)

    def test_load_gz(self):
        data = loadTensor("./live-tutorials/inputData/example_graph.tensor.gz", col_types=[int, int, int])
        self.assertIsInstance(data, TensorData)

    def test_load_csv(self):
        data = loadTensor("./live-tutorials/inputData/example_time_aggregate.csv")
        self.assertIsInstance(data, TensorData)

    def test_load_multi_file(self):
        data = loadTensor("./live-tutorials/inputData/yelp.edgelist")
        self.assertIsInstance(data, TensorData)
