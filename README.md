
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
| Type        | Abbr                                                                                             | Paper                                                                                                                                                                                                                                                                                              | Year         | Tutorials                                                                                  |
| :---------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------- | :----------------------------------------------------------------------------------------- |
| Time Series | [spartan.BeatLex](https://github.com/BGT-M/spartan2/tree/master/spartan/model/beatlex)           | [BEATLEX: Summarizing and Forecasting Time Series with Patterns](https://shenghua-liu.github.io/papers/pkdd2017-beatlex.pdf)                                                                                                                                                                       | 2017         | [BeatGAN](https://github.com/BGT-M/spartan2-tutorials/blob/master/BeatGAN.ipynb)           |
| Time Series | [spartan.BeatGAN](https://github.com/BGT-M/spartan2/tree/master/spartan/model/beatgan)           | [BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series](https://www.ijcai.org/Proceedings/2019/0616.pdf)<br>[Time Series Anomaly Detection with Adversarial Reconstruction Networks](https://ieeexplore.ieee.org/abstract/document/9669010/)                               | 2019<br>2022 | [Beatlex](https://github.com/BGT-M/spartan2-tutorials/blob/master/Beatlex.ipynb)           |
| Graph       | [spartan.HoloScope](https://github.com/BGT-M/spartan2/tree/master/spartan/model/holoscope)       | [HoloScope: Topology-and-Spike Aware Fraud Detection](https://shenghua-liu.github.io/papers/cikm2017-holoscope.pdf)<br>[A Contrast Metric for Fraud Detection in Rich Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8494803)                                                    | 2017<br>2019 | [HoloScope](https://github.com/BGT-M/spartan2-tutorials/blob/master/Holoscope.ipynb)       |
| Graph       | [spartan.Eigenspokes](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eigenspokes)   | [Eigenspokes: Surprising patterns and scalable community chipping in large graphs](https://www.cs.cmu.edu/~christos/PUBLICATIONS/pakdd10-eigenspokes.pdf)                                                                                                                                          | 2010         | [Eigenspokes](https://github.com/BGT-M/spartan2-tutorials/blob/master/EigenSpokes.ipynb)   |
| Graph       | [spartan.EagleMine](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eaglemine)       | [EagleMine: Vision-Guided Mining in Large Graphs](https://www.andrew.cmu.edu/user/lakoglu/odd/accepted_papers/ODD_v50_paper_25.pdf)                                                                                                                                                                | 2018         | [EagleMine](https://github.com/BGT-M/spartan2-tutorials/blob/master/EagleMine.ipynb)       |
| Graph       | [spartan.Fraudar](https://github.com/BGT-M/spartan2/tree/master/spartan/model/fraudar)           | [Fraudar: Bounding graph fraud in the face of camouflage](https://www.kdd.org/kdd2016/papers/files/rfp0110-hooiA.pdf)                                                                                                                                                                              | 2016         | [Fraudar](https://github.com/BGT-M/spartan2-tutorials/blob/master/Fraudar.ipynb)           |
| Graph       | [spartan.DPGS](https://github.com/BGT-M/spartan2/tree/master/spartan/model/DPGS)                 | [DPGS: Degree-Preserving Graph Summarization](https://shenghua-liu.github.io/papers/sdm2021-dpgs.pdf)                                                                                                                                                                                              | 2021         | [DPGS](https://github.com/BGT-M/spartan2-tutorials/blob/master/DPGS.ipynb)                 |
| Graph       | [spartan.EigenPulse](https://github.com/BGT-M/spartan2/tree/master/spartan/model/eigenpulse)     | [EigenPulse: Detecting Surges in Large Streaming Graphs with Row Augmentation](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_39)                                                                                                                                                     | 2019         | [EigenPulse](https://github.com/BGT-M/spartan2-tutorials/blob/master/EigenPulse.ipynb)     |
| Graph       | [spartan.FlowScope](https://github.com/BGT-M/spartan2/tree/master/spartan/model/flowscope)       | [FlowScope: Spotting Money Laundering Based on Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/5906)                                                                                                                                                                                      | 2020         | [FlowScope](https://github.com/BGT-M/spartan2-tutorials/blob/master/FlowScope.ipynb)       |
| Graph       | [spartan.kGrass](https://github.com/BGT-M/spartan2/tree/master/spartan/model/kGS)                | [GraSS: Graph structure summarization](https://ojs.aaai.org/index.php/AAAI/article/view/5906)                                                                                                                                                                                                      | 2010         | [kGrass](https://github.com/BGT-M/spartan2-tutorials/blob/master/kGrass.ipynb)             |
| Graph       | [spartan.IAT](https://github.com/BGT-M/spartan2/tree/master/spartan/model/iat)                   | [RSC: Mining and modeling temporal activity in social media](https://www.researchgate.net/profile/Alceu-Ferraz-Costa/publication/277311838_RSC_Mining_and_Modeling_Temporal_Activity_in_Social_Media/links/55945d2f08ae793d1379872f/RSC-Mining-and-Modeling-Temporal-Activity-in-Social-Media.pdf) | 2015         | [IAT](https://github.com/BGT-M/spartan2-tutorials/blob/master/iat_demo.ipynb)                   |
| Graph       | [spartan.CubeFlow](https://github.com/BGT-M/spartan2/tree/master/spartan/model/CubeFlow)         | [CubeFlow: Money Laundering Detection with Coupled Tensors](https://arxiv.org/pdf/2103.12411.pdf)                                                                                                                                                                                                  | 2021         | [CubeFlow](https://github.com/BGT-M/spartan2-tutorials/blob/master/CubeFlow.ipynb)         |
| Graph       | [spartan.Specgreedy](https://github.com/BGT-M/spartan2/tree/master/spartan/model/specgreedy)     |                                                                                                                                                                                                                                                                                                    |
| Graph       | [spartan.CubeFlowPlus](https://github.com/BGT-M/spartan2/tree/master/spartan/model/CubeFlowPlus) | [MonLAD: Money Laundering Agents Detection in Transaction Streams](https://arxiv.org/pdf/2201.10051.pdf)                                                                                                                                                                                           | 2022         | [CubeFlowPlus](https://github.com/BGT-M/spartan2-tutorials/blob/master/CubeFlowPlus.ipynb) |

## References
1. Bryan Hooi, Shenghua Liu, Asim Smailagic, and Christos Faloutsos, “BEATLEX: Summarizing and Forecasting Time Series with Patterns,” The European Conference on Machine Learning & Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), Skopje, Macedonia, 2017.
2. Zhou, Bin, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series." In IJCAI, pp. 4433-4439. 2019.
3. Liu, Shenghua, Bin Zhou, Quan Ding, Bryan Hooi, Zheng bo Zhang, Huawei Shen, and Xueqi Cheng. "Time Series Anomaly Detection with Adversarial Reconstruction Networks." IEEE Transactions on Knowledge and Data Engineering (2022).
4. Shenghua Liu, Bryan Hooi, and Christos Faloutsos, "HoloScope: Topology-and-Spike Aware Fraud Detection," In Proc. of ACM International Conference on Information and Knowledge Management (CIKM), Singapore, 2017, pp.1539-1548.
5. Shenghua Liu, Bryan Hooi, Christos Faloutsos, A Contrast Metric for Fraud Detection in Rich Graphs, IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol 31, Issue 12, Dec. 1 2019, pp. 2235-2248.
6. Prakash, B. Aditya, Ashwin Sridharan, Mukund Seshadri, Sridhar Machiraju, and Christos Faloutsos. "Eigenspokes: Surprising patterns and scalable community chipping in large graphs." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 435-448. Springer, Berlin, Heidelberg, 2010.
7. Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi, Huawei Shen, Xueqi Cheng, EagleMine: Vision-Guided Mining in Large Graphs, ACM SIGKDD 2018, ODD Workshop on Outlier Detection De-constructed, August 20th, London UK.
8. Hooi, Bryan, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. "Fraudar: Bounding graph fraud in the face of camouflage." In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 895-904. 2016.
9. Houquan Zhou, Shenghua Liu, Kyuhan Lee, Kijung Shin, Huawei Shen and Xueqi Cheng. "DPGS: Degree-Preserving Graph Summarization." In SDM, 2021.
10. Zhang, Jiabao, Shenghua Liu, Wenjian Yu, Wenjie Feng, and Xueqi Cheng. "Eigenpulse: Detecting surges in large streaming graphs with row augmentation." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 501-513. Springer, Cham, 2019.
11. Li, Xiangfeng, Shenghua Liu, Zifeng Li, Xiaotian Han, Chuan Shi, Bryan Hooi, He Huang, and Xueqi Cheng. "Flowscope: Spotting money laundering based on graphs." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 4731-4738. 2020.
12. LeFevre, Kristen, and Evimaria Terzi. "GraSS: Graph structure summarization." In Proceedings of the 2010 SIAM International Conference on Data Mining, pp. 454-465. Society for Industrial and Applied Mathematics, 2010.
13. Ferraz Costa, Alceu, Yuto Yamaguchi, Agma Juci Machado Traina, Caetano Traina Jr, and Christos Faloutsos. "Rsc: Mining and modeling temporal activity in social media." In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 269-278. 2015.
14. Sun, Xiaobing, Jiabao Zhang, Qiming Zhao, Shenghua Liu, Jinglei Chen, Ruoyu Zhuang, Huawei Shen, and Xueqi Cheng. "CubeFlow: Money Laundering Detection with Coupled Tensors." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 78-90. Springer, Cham, 2021.
15. Feng, Wenjie, Shenghua Liu, Danai Koutra, Huawei Shen, and Xueqi Cheng. "Specgreedy: unified dense subgraph detection." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 181-197. Springer, Cham, 2020.
16. Sun, Xiaobing, Wenjie Feng, Shenghua Liu, Yuyang Xie, Siddharth Bhatia, Bryan Hooi, Wenhan Wang, and Xueqi Cheng. "MonLAD: Money Laundering Agents Detection in Transaction Streams." arXiv preprint arXiv:2201.10051 (2022).
