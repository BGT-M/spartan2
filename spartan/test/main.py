#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Desc    :   Test entry point.
'''

# here put the import lib

import unittest
import os

s = unittest.TestSuite()
loader = unittest.TestLoader()
s.addTests(loader.discover(os.getcwd()))
run = unittest.TextTestRunner()
run.run(s)
