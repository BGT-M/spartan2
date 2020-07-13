`spartan` supports multiple computing backends, including `cpu`, and `gpu`.

By default, `cpu` is used. In this case, `numpy.ndarray` and `scipy.spmatrix` is used for dense tensor and sparse tensor respectively. When `gpu` is used, `torch.Tensor` is used. You can change it in config file (default position: `~/.spartan/config.json`).
```json
{
    "backend": "<backend_name>"
}
```
Setting the environment variable `SPARTAN_BACKEND` does the same, and it has higher priority than the config file.

When you import `spartan`, backend will be automatically loaded. You are allowed to dynamically change the backend by `spartan.load_backend()`.
```python
>>> import spartan as st
Using backend cpu
>>> st.load_backend('gpu')
Using backend gpu
```

You can check current backend by `st.get_backend()`:
```python
>>> import spartan as st
Using backend cpu
>>> st.get_backend()
'cpu'
>>> st.load_backend('gpu')
Using backend gpu
>>> st.get_backend()
'gpu'
```

In each backend, two classes `DTensor` and `STensor` are implemented for dense tensor and sparse tensor respectively. Several global apis are implemented (eg. `add()`, `dot()`) as well. For full supported apis, see file `spartan/backend/__init__.py`.

For spartan tensor usage example, see `tensor_usage.ipynb`.