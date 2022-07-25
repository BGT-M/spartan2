
# Welcome to spartan2

![](https://img.shields.io/badge/language-python-yellow.svg)
[![](https://img.shields.io/badge/pypi-0.1.3-brightgreen.svg)](https://pypi.org/project/spartan2/)
![](https://img.shields.io/github/forks/BGT-M/spartan2.svg?color=blue)
![](https://img.shields.io/github/stars/BGT-M/spartan2.svg?color=blue)
[![](https://readthedocs.org/projects/spartan2/badge/?version=latest)](https://spartan2.readthedocs.io/en/latest/)
[![](https://github.com/BGT-M/spartan2/actions/workflows/python-publish.yml/badge.svg)](https://github.com/BGT-M/spartan2/actions)
[![](https://img.shields.io/github/license/BGT-M/spartan2.svg)](https://github.com/BGT-M/spartan2/blob/master/LICENSE)

## Introduction

**spartan2** is a collection of data mining algorithms on **big graphs** and
**time series**, providing *three* basic tasks: ***anomaly detection***,
***forecast***, and ***summarization***. (see [readthedocs](https://spartan2.readthedocs.io/en/latest/), and [tutorials](https://github.com/BGT-M/spartan2-tutorials) )

Graphs and time series are fundamental representations of many key applications
in a wide range of 
- **online user behaviors**, e.g. *following in social media*, *shopping*, and *downloading Apps*,
- **finance**, e.g. *stock tradings, and bank transfers*, 
- **sensor networks**, e.g. *sensor readings, and smart power grid*,
and 
- **health**, e.g. *electrocardiogram, photoplethysmogram, and respiratory inductance plethysmography*.

In practice, we find that thinking graphs and time series as matrices or tensors
can enable us to find *efficient (near linear)*, *interpretable*, yet *accurate* solutions in many applications.
Therefore, our **goal** is developping a collectioin of algorithms on graphs and time series based
on **tensors** (matrix is a 2-mode tensor).

In real world, those tensors are *sparse*, and we
are required to make use of the sparsity to develop efficient algorithms. 
That is why we name the package as
**spartan**: **spar**se **t**ensor **an**alytics.

The package named **spartan** can be imported and run independently as a *usual python package*.
Everything in package **spartan** is viewed as a tensor (sparse).

## Install requirements

This project requires Python 3.7 and upper.
We suggest recreating the experimental environment using Anaconda through the following steps.

1. Install the appropriate version for Anaconda from here - https://www.anaconda.com/distribution/

2. Create a new conda environment named "spartan"
    ```bash
        conda create -n spartan python=3.7
        conda activate spartan
    ```

3. If you are a normal **USER**,
    ```bash
    # install spartan using pip
    pip install spartan2
    ```


4. If you want to **contribute**, or prefer to run directly on the code,
    <details>
        <summary>Please do the following setup</summary>

    - 4.1 Clone the project from github

        ```bash
        git clone git@github.com:BGT-M/spartan2.git
        ```

    - 4.2 Install requirements.
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

    - 4.3 Install code in development mode
        ```bash
        # in parent directory of spartan2
        pip install -e spartan2
        ```
    - 4.4 Since you install your package to a location other than the user site-packages directory, you will need to
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


## Table of Modules
| Type        | Abbr                                                                                             | Paper                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Year         | Tutorials                                                                                |
| :---------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------- | :--------------------------------------------------------------------------------------- |
| Graph       | [spartan.HoloScope](https://github.com/BGT-M/spartan2/tree/master/spartan/model/holoscope)       | [[1] HoloScope: Topology-and-Spike Aware Fraud Detection](#ref1) [[pdf]](https://shenghua-liu.github.io/papers/cikm2017-holoscope.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:togaDgTgsBkJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGsqJs:AAGBfm0AAAAAYl-qsJvi9t90zhgcm20QFQKe3In-ak_4&scisig=AAGBfm0AAAAAYl-qsM3_hRgnbhToH1xl6vPPvAqAWWLW&scisf=4&ct=citation&cd=-1&hl=zh-CN)<br>[[2] A Contrast Metric for Fraud Detection in Rich Graphs](#ref2) [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8494803) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:9ti0-P3zZgUJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGtTfw:AAGBfm0AAAAAYl-rVfy-DhMFowBZIcvf2j6FxHdf9SG0&scisig=AAGBfm0AAAAAYl-rVXIObYCkVW7zzyvR2vsg82rxVu6_&scisf=4&ct=citation&cd=-1&hl=zh-CN)                          | 2017<br>2019 | [HoloScope](https://github.com/BGT-M/spartan2-tutorials/blob/master/Holoscope.ipynb)     |
| Graph       | [spartan.Eigenspokes](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eigenspokes)   | [[3] Eigenspokes: Surprising patterns and scalable community chipping in large graphs](#ref3) [[pdf]](https://www.cs.cmu.edu/~christos/PUBLICATIONS/pakdd10-eigenspokes.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:qBJlLWkeAfMJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGt2fg:AAGBfm0AAAAAYl-rwfiY3cBc4sbe0DQ1zI_5vOZKZvsL&scisig=AAGBfm0AAAAAYl-rwct94h11DB0JGGOlhtyVDzOQAmGu&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                     | 2010         | [Eigenspokes](https://github.com/BGT-M/spartan2-tutorials/blob/master/EigenSpokes.ipynb) |
| Graph       | [spartan.EagleMine](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eaglemine)       | [[4] EagleMine: Vision-guided Micro-clusters recognition and collective anomaly detection](#ref4) [[pdf]](https://shenghua-liu.github.io/papers/FGCS2021-eaglemine.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:ZmPXgb_5mPkJ:scholar.google.com/&output=citation&scisdr=CgWn_i8VELHrwZFKzN0:AAGBfm0AAAAAYmtP1N0ivIme2D7w_PpQzjQM0RrCiupy&scisig=AAGBfm0AAAAAYmtP1AdwcwDSfhqh4qF0K9GQ4d5BlUny&scisf=4&ct=citation&cd=-1&hl=en)  <br> [ Beyond outliers and on to micro-clusters: Vision-guided anomaly detection ](#ref4.2) [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_42) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:4oBxXcPakYYJ:scholar.google.com/&output=citation&scisdr=CgWn_i8VELHrwZFXGyM:AAGBfm0AAAAAYmtSAyOFZbSm7Ykzu0wLEaHiH5dEg_D5&scisig=AAGBfm0AAAAAYmtSAxPgpRdIquO8yiF4Bdzv-LYmjVoW&scisf=4&ct=citation&cd=-1&hl=en)                                                                                                                                                                                                                                                                                                                                                                                                                                      | 2021<br>2019         | [EagleMine](https://github.com/BGT-M/spartan2-tutorials/blob/master/EagleMine.ipynb)     |
| Graph       | [spartan.Fraudar](https://github.com/BGT-M/spartan2/tree/master/spartan/model/fraudar)           | [[5] Fraudar: Bounding graph fraud in the face of camouflage](#ref5) [[pdf]](https://www.kdd.org/kdd2016/papers/files/rfp0110-hooiA.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:E4UJrtgtBg8J:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGt9Xc:AAGBfm0AAAAAYl-r7Xd-6mLNmUbQxg1uayrYJlURHVOp&scisig=AAGBfm0AAAAAYl-r7ZEcoy8XyrI0yhyABnm50HdYj0K_&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 2016         | [Fraudar](https://github.com/BGT-M/spartan2-tutorials/blob/master/Fraudar.ipynb)         |
| Graph       | [spartan.DPGS](https://github.com/BGT-M/spartan2/tree/master/spartan/model/DPGS)                 | [[6] DPGS: Degree-Preserving Graph Summarization](#ref6) [[pdf]](https://shenghua-liu.github.io/papers/sdm2021-dpgs.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:sU5z3lgSGbQJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqHZY:AAGBfm0AAAAAYl-sBZaRFQmrJx5N_kAREk-gKm7YYNQa&scisig=AAGBfm0AAAAAYl-sBe3YdPWS2HKGLhYpRWVXVfEH_vx3&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 2021         | [DPGS](https://github.com/BGT-M/spartan2-tutorials/blob/master/DPGS.ipynb)               |
| Graph       | [spartan.EigenPulse](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eigenpulse)     | [[7] EigenPulse: Detecting Surges in Large Streaming Graphs with Row Augmentation](#ref7) [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_39) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:9aY1PsE_nZQJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqDxA:AAGBfm0AAAAAYl-sFxCgazegMNgTpiRIfnS3yTB2KylB&scisig=AAGBfm0AAAAAYl-sF6gcDlB_MvXV40kKvjGAnfG2BIgr&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                | 2019         | [EigenPulse](https://github.com/BGT-M/spartan2-tutorials/blob/master/EigenPulse.ipynb)   |
| Graph       | [spartan.FlowScope](https://github.com/BGT-M/spartan2/tree/master/spartan/model/flowscope)       | [[8] FlowScope: Spotting Money Laundering Based on Graphs](#ref8) [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/5906) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:ISC57gjxaOoJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqN-8:AAGBfm0AAAAAYl-sL-9q0VZ0BrLYhf2l74rXRQoP8cuE&scisig=AAGBfm0AAAAAYl-sL-jeaAspS90ZoR9rxklbdNUCiSIm&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 2020         | [FlowScope](https://github.com/BGT-M/spartan2-tutorials/blob/master/FlowScope.ipynb)     |
| Graph       | [spartan.kGrass](https://github.com/BGT-M/spartan2/tree/master/spartan/model/kGS)                | [[9] GraSS: Graph structure summarization](#ref9) [[pdf]](http://cs-people.bu.edu/evimaria/papers/Social-net.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:9EOa6QuDHZAJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqWMA:AAGBfm0AAAAAYl-sQMD14UtTxhKsp_oA7zSIRy-WadA2&scisig=AAGBfm0AAAAAYl-sQAjwBpfLFRBFBnaUfLl-bmfRGHGW&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 2010         | [kGrass](https://github.com/BGT-M/spartan2-tutorials/blob/master/kGrass.ipynb)           |
| Graph       | [spartan.IAT](https://github.com/BGT-M/spartan2/tree/master/spartan/model/iat)                   | [[10] RSC: Mining and modeling temporal activity in social media](#ref10) [[pdf]](https://www.researchgate.net/profile/Alceu-Ferraz-Costa/publication/277311838_RSC_Mining_and_Modeling_Temporal_Activity_in_Social_Media/links/55945d2f08ae793d1379872f/RSC-Mining-and-Modeling-Temporal-Activity-in-Social-Media.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:Y5foKpCdwI0J:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqRdo:AAGBfm0AAAAAYl-sXdqwuLOSp8pBWvdIWdIar1cvgRXw&scisig=AAGBfm0AAAAAYl-sXY0yOktkYEcOeKED3TJiYjFYxv9-&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                          | 2015         | [IAT](https://github.com/BGT-M/spartan2-tutorials/blob/master/iat_demo.ipynb)            |
| Graph       | [spartan.CubeFlow](https://github.com/BGT-M/spartan2/tree/master/spartan/model/CubeFlow)         | [[11] CubeFlow: Money Laundering Detection with Coupled Tensors](#ref11) [[pdf]](https://arxiv.org/pdf/2103.12411.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:G4N98aANZ5gJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqZM4:AAGBfm0AAAAAYl-sfM5bxT4PH7UnPWisGJ_GqiyJaHtA&scisig=AAGBfm0AAAAAYl-sfPaRdTlmv1K-LEIjcEPbDXsVWqRh&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 2021         | [CubeFlow](https://github.com/BGT-M/spartan2-tutorials/blob/master/CubeFlow.ipynb)       |
| Graph       | [spartan.SpecGreedy](https://github.com/BGT-M/spartan2/tree/master/spartan/model/SpecGreedy)     | [[12] Specgreedy: unified dense subgraph detection](#ref12) [[pdf]](https://shenghua-liu.github.io/papers/pkdd2020_specgreedy.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:Zkz3wV1GTDEJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqlq0:AAGBfm0AAAAAYl-sjq1OnEb4MYGD5IRKP4ZHfqBZ2WRS&scisig=AAGBfm0AAAAAYl-sjo13h7RhyfCbKY6bJUIT804zBoLl&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 2020         | [SpecGreedy](https://github.com/BGT-M/spartan2-tutorials/blob/master/SpecGreedy.ipynb)   |
| Graph | [spartan.MonLAD](https://github.com/BGT-M/spartan2/tree/master/spartan/model/MonLAD) | [[13] MonLAD: Money Laundering Agents Detection in Transaction Streams](#ref13) [[pdf]](https://shenghua-liu.github.io/papers/wsdm2022-monlad.pdf) | 2022 |[MonLAD]()|
| Time Series | [spartan.BeatLex](https://github.com/BGT-M/spartan2/tree/master/spartan/model/beatlex)           | [[14] BEATLEX: Summarizing and Forecasting Time Series with Patterns](#ref14) [[pdf]](https://shenghua-liu.github.io/papers/pkdd2017-beatlex.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:pLIN80jhGHgJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGq0d8:AAGBfm0AAAAAYl-syd9yVZFzWiYMiHdbUG0Otq_40L_4&scisig=AAGBfm0AAAAAYl-syQvhHWrRiDudGxesup58a12Rb2fj&scisf=4&ct=citation&cd=-1&hl=zh-CN)                                                                                                                                                                                                                                                                                                                                                                                                                                                | 2017         | [Beatlex](https://github.com/BGT-M/spartan2-tutorials/blob/master/Beatlex.ipynb)         |
| Time Series | [spartan.BeatGAN](https://github.com/BGT-M/spartan2/tree/master/spartan/model/beatgan)           | [[15] BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series](#ref15) [[pdf]](https://www.ijcai.org/Proceedings/2019/0616.pdf) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:GWv3XJMw-f8J:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGqwVQ:AAGBfm0AAAAAYl-s2VSO3uxiyQMF9ttr8Cvvwj2034HZ&scisig=AAGBfm0AAAAAYl-s2Vv9GFSCyVpThmaCvfxrxGjIuU6L&scisf=4&ct=citation&cd=-1&hl=zh-CN)<br>[[16] Time Series Anomaly Detection with Adversarial Reconstruction Networks](#ref16) [[pdf]](https://ieeexplore.ieee.org/abstract/document/9669010/) [[bib]](https://scholar.googleusercontent.com/scholar.bib?q=info:o3Pi94N8dRAJ:scholar.google.com/&output=citation&scisdr=CgVSaNk4EIGjstGq_0M:AAGBfm0AAAAAYl-s50Orj2CzeO_ef9IUAjpK3Q5xFBuk&scisig=AAGBfm0AAAAAYl-s56FvzCW7mHJ5VTdWoDSDQF8HsvTu&scisf=4&ct=citation&cd=-1&hl=zh-CN) | 2019<br>2022 | [BeatGAN](https://github.com/BGT-M/spartan2-tutorials/blob/master/BeatGAN.ipynb)         |

## References
1. <span id="ref1"></span> Shenghua Liu, Bryan Hooi, and Christos Faloutsos, "HoloScope: Topology-and-Spike Aware Fraud Detection," In Proc. of ACM International Conference on Information and Knowledge Management (CIKM), Singapore, 2017, pp.1539-1548.
2. <span id="ref2"></span> Shenghua Liu, Bryan Hooi, Christos Faloutsos, A Contrast Metric for Fraud Detection in Rich Graphs, IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol 31, Issue 12, Dec. 1 2019, pp. 2235-2248.
3. <span id="ref3"></span> Prakash, B. Aditya, Ashwin Sridharan, Mukund Seshadri, Sridhar Machiraju, and Christos Faloutsos. "Eigenspokes: Surprising patterns and scalable community chipping in large graphs." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 435-448. Springer, Berlin, Heidelberg, 2010.
4. <span id="ref4"></span> Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi, Huawei Shen, and Xueqi Cheng. EagleMine: Vision-guided Micro-clusters recognition and collective anomaly detection, Future Generation Computer Systems, Vol 115, Feb 2021, pp.236-250.
   
   <span id="ref4.2"></span>Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi, Huawei Shen, Xueqi Cheng, Beyond outliers and on to micro-clusters: Vision-guided anomaly detection, In Proc. of the 23rd Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2019), 2019, Macau, China, pp541-554.
5. <span id="ref5"></span> Hooi, Bryan, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. "Fraudar: Bounding graph fraud in the face of camouflage." In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 895-904. 2016.
6. <span id="ref6"></span> Houquan Zhou, Shenghua Liu, Kyuhan Lee, Kijung Shin, Huawei Shen and Xueqi Cheng. "DPGS: Degree-Preserving Graph Summarization." In SDM, 2021.
7. <span id="ref7"></span> Zhang, Jiabao, Shenghua Liu, Wenjian Yu, Wenjie Feng, and Xueqi Cheng. "Eigenpulse: Detecting surges in large streaming graphs with row augmentation." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 501-513. Springer, Cham, 2019.
8. <span id="ref8"></span> Li, Xiangfeng, Shenghua Liu, Zifeng Li, Xiaotian Han, Chuan Shi, Bryan Hooi, He Huang, and Xueqi Cheng. "Flowscope: Spotting money laundering based on graphs." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 4731-4738. 2020.
9. <span id="ref9"></span> LeFevre, Kristen, and Evimaria Terzi. "GraSS: Graph structure summarization." In Proceedings of the 2010 SIAM International Conference on Data Mining, pp. 454-465. Society for Industrial and Applied Mathematics, 2010.
10. <span id="ref10"></span> Ferraz Costa, Alceu, Yuto Yamaguchi, Agma Juci Machado Traina, Caetano Traina Jr, and Christos Faloutsos. "Rsc: Mining and modeling temporal activity in social media." In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 269-278. 2015.
11. <span id="ref11"></span> Sun, Xiaobing, Jiabao Zhang, Qiming Zhao, Shenghua Liu, Jinglei Chen, Ruoyu Zhuang, Huawei Shen, and Xueqi Cheng. "CubeFlow: Money Laundering Detection with Coupled Tensors." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 78-90. Springer, Cham, 2021.
12. <span id="ref12"></span> Feng, Wenjie, Shenghua Liu, Danai Koutra, Huawei Shen, and Xueqi Cheng. "Specgreedy: unified dense subgraph detection." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 181-197. Springer, Cham, 2020.
13. <span id="ref13"></span> Sun, Xiaobing, Wenjie Feng, Shenghua Liu, Yuyang Xie, Siddharth Bhatia, Bryan Hooi, Wenhan Wang, and Xueqi Cheng. "MonLAD: Money Laundering Agents Detection in Transaction Streams." arXiv preprint arXiv:2201.10051 (2022).
14. <span id="ref14"></span> Bryan Hooi, Shenghua Liu, Asim Smailagic, and Christos Faloutsos, “BEATLEX: Summarizing and Forecasting Time Series with Patterns,” The European Conference on Machine Learning & Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), Skopje, Macedonia, 2017.
15. <span id="ref15"></span> Zhou, Bin, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series." In IJCAI, pp. 4433-4439. 2019.
16. <span id="ref16"></span> Liu, Shenghua, Bin Zhou, Quan Ding, Bryan Hooi, Zheng bo Zhang, Huawei Shen, and Xueqi Cheng. "Time Series Anomaly Detection with Adversarial Reconstruction Networks." IEEE Transactions on Knowledge and Data Engineering (2022).
