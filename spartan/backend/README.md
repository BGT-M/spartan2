`spartan` supports multiple computing backends, including `numpy`, `scipy` and `pytorch`.

By default, `numpy` is used. You can change it in config file (default position: `~/.spartan/config.json`).
```json
{
    "backend": "<backend_name>"
}
```
Setting the environment varaiable `SPARTAN_BACKEND` does the same, and it has higher priority than the config file.

When you import `spartan`, backend will be automatically loaded. You can call backend's apis by `spartan.<api>`:
```python
>>>import spartan as st
Using backend numpy
>>>st.add == np.add
True
```
For full supported apis, see file `spartan/backend/__init__.py`.

You are allowed to dynamically change the backend by `spartan.load_backend()`.
```python
>>> import spartan as st
Using backend numpy
>>> import numpy as np
>>> st.add == np.add
True
>>> st.load_backend('scipy')
Using backend scipy
>>> import scipy as sp
>>> st.add == sp.add
True
>>> st.load_backend('pytorch')
Using backend pytorch
>>> import torch
>>> st.add == torch.add
True
```

You can check current backend by `st.backend()`:
```python
>>> import spartan as st
Using backend numpy
>>> st.backend()
'numpy'
>>> st.load_backend('pytorch')
Using backend pytorch
>>> st.backend()
'pytorch'
```