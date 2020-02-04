# spartan2: backend for SparTan
This is a developping version for our future SparTan.
SparTan: Sparse Tensor Analytics
Everthing here is viewed as a tensor (sparse).

Interpreter Version: **Python 3.6**

*Follow steps below to run the project*:

1. Open terminal and Enter the demo directory of this project
2. Run Command：``python test.****.py``

# install requires
```bash
pip3 install -r requirements
```

# API Usage:
------

```python

import spartan as st

# set the computing engine
st.config(st.engine.SINGLEMACHINE)

# load graph data, data stores as edgelist in database, e.g.~pandas, sqlite, postgress, hive
data = st.loadTensor(name="yelp", path="~/Data/", col_ids = ["uid", "oid", "rating"], col_types = [int, int, int])

```
## count triangles:

```python
 # create triangle count model
trimodel = st.triangle_count.create(data, st.tc_policy.DOULION, "my_doulion_model")

 # run the model by default set undirected for graphs
trimodel.run(p=0.8)

 # show the results
trimodel.showResults(plot=True)
```

## anomaly detection
```python
 # create a anomaly detection model
hsmodel = st.anomaly_detection.create(data, st.ad_policy.HOLOSCOPE, "my_holoscope_model")

 # run the model
hsmodel.run(k=3)

 # show the results
hsmodel.showResults()
```

## eigen decomposition
```python
# create a eigen decomposition model
edmodel = st.eigen_decompose.create(data, st.ed_policy.SVDS, "my_svds_model")

# run the model
edmodel.run(k=10)

# show the result
edmodel.showResults()
```

## degree
```python
# count degree
Du, Dv = st.bidegree(data)
# D = st.degree(data)
```

## EagleMine
```python
 # create a anomaly detection model
emmodel = st.anomaly_detection.create(data, st.ad_policy.EAGLEMINE, "my_eaglemine_model")
emmodel.setbipartite(True)
 # run the eaglemine model
emmodel.run(edmodel.U, Du)

emmodel.run(edmodel.V, Dv)

A, B = emmodel.nodes(n=0)

```
## subgraph
```python
g = st.subgraph(data, A, B)
# g = st.subgraph(data, A)
```



## Table of Contents

**Part 1: Basic usage**

* [ioutil](https://github.com/shenghua-liu/spartan2/blob/master/ioutil_demo.ipynb)
* [Time series data](https://github.com/shenghua-liu/spartan2/blob/master/TimeseriesData_demo.ipynb)

**Part 2: Demo**

* [SVD](https://github.com/shenghua-liu/spartan2/blob/master/SVD_demo.ipynb)

* [Beatlex](https://github.com/shenghua-liu/spartan2/blob/master/Beatlex_demo.ipynb)
* [Eaglemine](https://github.com/shenghua-liu/spartan2/blob/master/Eaglemine_demo.ipynb)
* [Fraudar](https://github.com/shenghua-liu/spartan2/blob/master/Fraudar_demo.ipynb)
* [Holoscope](https://github.com/shenghua-liu/spartan2/blob/master/Holoscope_demo.ipynb)