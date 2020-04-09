from setuptools import setup

NAME = 'spartan2'
VERSION = 0
DESCRIPTION = 'collection of data mining algorithms on big graphs and time series'
URL = 'https://github.com/shenghua-liu/spartan2'
AUTHOR = 'Shenghua Liu, Houquan Zhou, Quan Ding'
EMAIL = 'liu.shengh@foxmail.com'

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
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      install_requires=REQUIRED_PACKAGES,
      packages=['spartan2', 'spartan2.algorithm', 'spartan2.models', 'spartan2.tensor'],
      )
