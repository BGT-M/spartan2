
# Welcome to spartan2

## introduction

**spartan2** is a collection of data mining algorithms on **big graphs** and
**time series**, providing *three* basic tasks: ***anomaly detection***,
***forecast***, and ***summarization***. (see [readthedocs](https://spartan2.readthedocs.io/en/latest/), and [tutorials](https://github.com/BGT-M/spartan2-tutorials) )

Graphs and time series are fundamental representations of many key applications
in a wide range of users' online behaviors (e.g. social media, shopping, Apps),
finance (e.g. stock tradings, bank transfers), IoT networks (e.g. sensor readings, smart power grid),
and healthcare (e.g. electrocardiogram, photoplethysmogram, respiratory inductance plethysmography).
In practice, we find that thinking graphs and time series as matrices or tensors
can enable us to find *efficient (near linear)*, *interpretable*, yet *accurate* solutions in many applications.

Therefore, our **goal** is developping a collectioin of algorithms on graphs and time series based
on **tensors** (matrix is a 2-mode tensor).

In real world, those tensors are *sparse*, and we
are required to make use of the sparsity to develop efficient algorithms. That is why
we name the backend package and a private UI interface as

**SparTAn**: **Spar**se **T**ensor **An**alytics.

spartan2 is the backend name. The package named **spartan** can be imported and run independently as a *usual python package*.
Everything in package **spartan** is viewed as a tensor (sparse).

## install requirements

This project requires Python 3.7 and upper.
We suggest recreating the experimental environment using Anaconda through the following steps.

1. Install the appropriate version for Anaconda from here - https://www.anaconda.com/distribution/

2. Create a new conda environment named "spartan"
```bash
conda create -n spartan python=3.7
conda activate spartan
```

3. If you are a normal **USER**,
<details><summary>download the package from pip</summary>

```bash
pip install spartan2
```

</details>


4. If you are a **DEVELOPER** and **contribute** to the project, or prefer to run directly on the code,
<details>
    <summary>please do the following setup</summary>

4.1 Clone the project from github

```bash
git clone https://github.com/shenghua-liu/spartan2.git
```

4.2 Install requirements.
```bash
# [not recommended]# pip install --user --requirement requirements
# using conda tool
conda install --force-reinstall -y --name spartan -c conda-forge --file requirements
```

*or use the following way*

```bash
# this may not work in ubuntu 18.04
python setup.py install
```

4.3 Install code in development mode
```bash
# in parent directory of spartan2
pip install -e spartan2
```
4.4 Since you install your package to a location other than the user site-packages directory, you will need to
add environment variable PYTHONPATH in ~/.bashrc

```bash
export PYTHONPATH=/<dir to spartan2>/spartan2:$PYTHONPATH
```

*or prepend the path to that directory to your PYTHONPATH environment variable.*

```python
import sys
sys.path.append("/<dir to spartan2>/spartan2")
```
*or do as follows*

```bash
#find directory of site-packages
python -c 'import site; print(site.getsitepackages())'

#add \<name\>.pth file in your site-packages directory with string '/<dir to spartan2>/spartan2'

```

</details>


5. start jupyter notebook, and try live tutorials for demos:
<details><summary><strong>live-tutorials</strong></summary>

**Table of Contents**

All contents are collected in another repository [spartan-tutorials](https://github.com/BGT-M/spartan2-tutorials), you can clone that repository to get all the notebooks and example data to run on your own.

**Part 1: Basic**
* [Quick start](https://github.com/BGT-M/spartan2-tutorials/blob/master/quick_start.ipynb)
* [Tensor usage](https://github.com/BGT-M/spartan2-tutorials/blob/master/tensor_usage.ipynb)

**Part 2: Big Graphs**
* [Graph start](https://github.com/BGT-M/spartan2-tutorials/blob/master/graph_start.ipynb)
* [SpokEn](https://github.com/BGT-M/spartan2-tutorials/blob/master/SVD_demo.ipynb): an implementation of [EigenSpokes](http://www.cs.cmu.edu/~christos/PUBLICATIONS/pakdd10-eigenspokes.pdf) by SVD.
* [Eaglemine](https://github.com/BGT-M/spartan2-tutorials/blob/master/EagleMine.ipynb)
* [Fraudar](https://github.com/BGT-M/spartan2-tutorials/blob/master/Fraudar_demo.ipynb): a wrapper of [Fraudar](https://bhooi.github.io/projects/fraudar/index.html) algorithm.
* [Holoscope](https://github.com/BGT-M/spartan2-tutorials/blob/master/Holoscope.ipynb): based on [HoloScope](https://shenghua-liu.github.io/papers/cikm2017-holoscope.pdf)
* [EigenPulse](https://github.com/BGT-M/spartan2-tutorials/blob/master/EigenPulse.ipynb)
* [DPGS](https://github.com/BGT-M/spartan2-tutorials/blob/master/DPGS.ipynb)

**Part 3: Time Series**
* [Time Series start](https://github.com/BGT-M/spartan2-tutorials/blob/master/timeseries_start.ipynb)
* [Other operations](https://github.com/BGT-M/spartan2-tutorials/blob/master/Log2Timeseries.ipynb)
* [Beatlex](https://github.com/BGT-M/spartan2-tutorials/blob/master/Beatlex.ipynb): based on [BeatLex](https://shenghua-liu.github.io/papers/pkdd2017-beatlex.pdf)
* [BeatGAN](https://github.com/BGT-M/spartan2-tutorials/blob/master/BeatGAN.ipynb): based on [BeatGAN](https://www.ijcai.org/Proceedings/2019/0616.pdf)

</details>

## API docs

For more details to use spartan2, please see the api docs [readthedocs](https://spartan2.readthedocs.io/en/latest/).

## references
1. Shenghua Liu, Bryan Hooi, Christos Faloutsos, A Contrast Metric for Fraud Detection in Rich Graphs, IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol 31, Issue 12, Dec. 1 2019, pp. 2235-2248.
1. Shenghua Liu, Bryan Hooi, and Christos Faloutsos, "HoloScope: Topology-and-Spike Aware Fraud Detection," In Proc. of ACM International Conference on Information and Knowledge Management (CIKM), Singapore, 2017, pp.1539-1548.
2. Prakash, B. Aditya, Ashwin Sridharan, Mukund Seshadri, Sridhar Machiraju, and Christos Faloutsos. "Eigenspokes: Surprising patterns and scalable community chipping in large graphs." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 435-448. Springer, Berlin, Heidelberg, 2010.
3. Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi, Huawei Shen, Xueqi Cheng, EagleMine: Vision-Guided Mining in Large Graphs, ACM SIGKDD 2018, ODD Workshop on Outlier Detection De-constructed, August 20th, London UK.
4. Bryan Hooi, Shenghua Liu, Asim Smailagic, and Christos Faloutsos, “BEATLEX: Summarizing and Forecasting Time Series with Patterns,” The European Conference on Machine Learning & Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), Skopje, Macedonia, 2017.
5. Hooi, Bryan, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. "Fraudar: Bounding graph fraud in the face of camouflage." In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 895-904. 2016.
6. Zhou, Bin, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series." In IJCAI, pp. 4433-4439. 2019.
7. Houquan Zhou, Shenghua Liu, Kyuhan Lee, Kijung Shin, Huawei Shen and Xueqi Cheng. "DPGS: Degree-Preserving Graph Summarization." In SDM, 2021.
