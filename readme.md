# spartan2:

**spartan2** is a collection of data mining algorithms on **big graphs** and
**time series**, as graphs and time series are fundamental representations of many key applications 
in a wide range of users' online behaviors (e.g. social media, shopping, Apps), 
finance (e.g. stock tradings, bank transfers), IoT networks (e.g. sensor readings, smart power grid), and healthcare (e.g. electrocardiogram, photoplethysmogram, respiratory inductance plethysmography). 

In practice, we find that thinking graphs and time series as matrices or tensors
can enable us to find efficient (near linear), interpretable, yet accurate solutions in many applications.

Therefore, we want to develop a collectioin of algorithms on graphs and time series based
on tensors (matrix is a 2-mode tensor). In real world, those tensors are sparse, and we
are required to make use of the sparsity to develop efficient algorithms. That is why
we name the package of algorithms as 

**SparTan**: **Spar**se **T**ensor **An**alytics.

spartan2 is backend of SparTAn.
Everthing here is viewed as a tensor (sparse).

Interpreter Version: **Python 3.6** and above.

## install requires
```bash
pip3 install -r requirements
```


## Follow steps below to run the project demo:

1. start jupyter notebook
2. click to see each jupyter notebook (xxx.ipynb) demo for each algorithm, or see the following guidline.


## Table of Contents

**Part 1: Basic**
* [Quick start](https://github.com/shenghua-liu/spartan2/blob/master/quick_start.ipynb)


**Part 2: Big Graphs**
* [Load graph](https://github.com/shenghua-liu/spartan2/blob/master/ioutil_demo.ipynb)
* [SVD](https://github.com/shenghua-liu/spartan2/blob/master/SVD_demo.ipynb)
* [Eaglemine](https://github.com/shenghua-liu/spartan2/blob/master/Eaglemine_demo.ipynb)
* [Fraudar](https://github.com/shenghua-liu/spartan2/blob/master/Fraudar_demo.ipynb)
* [Holoscope](https://github.com/shenghua-liu/spartan2/blob/master/Holoscope_demo.ipynb)

**Part 3: Time Series**
* [Load time series](https://github.com/shenghua-liu/spartan2/blob/master/TimeseriesData_demo.ipynb)
* [Beatlex](https://github.com/shenghua-liu/spartan2/blob/master/Beatlex_demo.ipynb)
