import unittest
import os

s = unittest.TestSuite()
loader = unittest.TestLoader()
s.addTests(loader.discover(os.getcwd()))
run = unittest.TextTestRunner()
run.run(s)
