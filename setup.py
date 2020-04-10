# Copyright 2020 The spartan2 Authors.
#
#

from setuptools import setup

NAME = 'spartan2'
VERSION = 0
DESCRIPTION = 'collection of data mining algorithms on big graphs and time series'
URL = 'https://github.com/shenghua-liu/spartan2'
AUTHOR = 'Shenghua Liu, Houquan Zhou, Quan Ding'
EMAIL = 'liu.shengh@foxmail.com'

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()
  
  
REQUIRED_PACKAGES = [
      'numpy',
      'scipy',
      'networkx',
      'matplotlib',
      'statsmodels',
      'pomegranate',
      'scikit-learn',
      'scikit-image',
]

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=_LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      install_requires=REQUIRED_PACKAGES,
      packages=['spartan2', 'spartan2.algorithm', 'spartan2.models', 'spartan2.tensor'],
      )
