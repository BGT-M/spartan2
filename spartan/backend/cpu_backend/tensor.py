import functools

import numpy as np
import scipy.sparse as ssp


def _wrap_ret(squeeze=False):
    """Wrap return value of func to spartan tensor types.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if squeeze:
                if isinstance(ret, np.matrix):
                    ret = ret.A.squeeze()
            if isinstance(ret, np.ndarray):
                return DTensor(ret)
            elif ssp.isspmatrix(ret):
                return STensor(ret)
            else:
                return ret
        return wrapper
    return decorator


def _require_dense(*pos):
    """Ensure parameters at pos are DTensor.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            params = [args[p] for p in pos]
            for p, param in zip(pos, params):
                if not isinstance(param, DTensor):
                    raise TypeError(
                        f"The {p}-th parameter of `st.{func.__name__}` does not support {type(param)} type.")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_params(*pos):
    """Ensure parameters at pos are same spartan types.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            params = [args[p] for p in pos]
            is_sparse = all([isinstance(p, STensor) for p in params])
            is_dense = all([isinstance(p, DTensor) for p in params])
            if not (is_sparse or is_dense):
                types = [str(type(p)) for p in params]
                msg = f"Unsupported type for `st.{func.__name__}`: {', '.join(types)}"
                raise TypeError(msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class STensor:
    def __init__(self, data, dtype=None, shape=None, format='coo'):
        if ssp.isspmatrix(data):
            self._data = data
        else:
            self._data = ssp.coo_matrix(
                data, shape=shape, dtype=dtype).asformat(format)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    @_wrap_ret()
    def __getattr__(self, name):
        ret = getattr(self._data, name)
        if callable(ret):
            return _wrap_ret(True)(ret)
        return ret

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._data, name, value)

    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            delattr(self._data, name)


class DTensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data, dtype=dtype)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    @_wrap_ret()
    def __getattr__(self, name):
        ret = getattr(self._data, name)
        if callable(ret):
            return _wrap_ret(True)(ret)
        return ret

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._data, name, value)

    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            delattr(self._data, name)
