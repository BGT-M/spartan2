# PocForSpartan
This is a POC for our future Spartan

Interpreter Version: **Python 2.7**

*Follow steps below to run the project*:

1. Open terminal and Enter the demo directory of this project
2. Run Command：``python test.****.py``

# API Usage:
------

```python

import spartan as st
   
# set the computing engine
st.config(st.engine.SINGLEMACHINE)

# load graph data
data = st.loadTensor(name="yelp", path="~/Data/", schema=("uid":str, "oid":str, "ts":int, "rating":float))
```
## count triangles:

```python
 # create triangle count model
trimodel = st.triangle_count.create(data, "triangle count")

 # run the model 
trimodel.run(st.tc_policy.DOULION, p=0.8)

 # show the results
trimodel.showResults(plot=True)
```

## anomaly detection
```python
 # create a anomaly detection model
admodel = st.anomaly_detection.create(data, "anomaly detection")

 # run the model
admodel.run(st.ad_policy.HOLOSCOPE, k=3)

 # show the results
model.showResults()
```

## eigen decomposition
```python
# create a eigen decomposition model
edmodel = st.eigen_decompose.create(data, "eigen decomposition")

# run the model
edmodel.run(st.ed_policy.SVDS, k=10)

# show the result
edmodel.showResults()
```
