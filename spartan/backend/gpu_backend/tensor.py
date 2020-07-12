import functools
import numbers
from numpy.lib.arraysetops import isin

import torch
import torch.sparse as sparse


def _wrap_ret():
    """Wrap return value of func to spartan tensor types.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if isinstance(ret, torch.Tensor):
                if ret.ndim == 0:
                    return ret.item()
                if ret.is_sparse:
                    return STensor(ret)
                else:
                    return DTensor(ret)
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
                msg = f"Unsupported type in `st.{func.__name__}`: {', '.join(types)}"
                raise TypeError(msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class DTensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, DTensor):
            self._data = data._data
        elif isinstance(data, STensor):
            self._data = data._data.to_dense()
        else:
            self._data = torch.as_tensor(data, dtype=dtype)
        self._data = self._data.cuda()

    def __repr__(self):
        return '%s(\n%r\n)' % (type(self).__name__, self._data)

    def __len__(self):
        return self._data.__len__()

    # Slice
    @_wrap_ret()
    def __getitem__(self, index):
        return self._data.__getitem__(index)

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def __delitem__(self, index):
        self._data.__delitem__(index)

    @_wrap_ret()
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        ret = getattr(self._data, name)
        if callable(ret):
            raise AttributeError(
                f"{type(self).__name__} doesn't have attribute {name}")
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


class STensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, DTensor):
            self._data = data._data.to_sparse()
        elif isinstance(data, STensor):
            self._data = data._data
        else:
            self._data = torch.as_tensor(data, dtype=dtype).to_sparse()
        self._data = self._data.cuda()

    def __repr__(self):
        return '%s(\n%r\n)' % (type(self).__name__, self._data)

    def __len__(self):
        return self._data.__len__()

    # Pytorch's GPU sparse tensor doesn't support slice yet!

    @_wrap_ret()
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        ret = getattr(self._data, name)
        if callable(ret):
            raise AttributeError(
                f"{type(self).__name__} doesn't have attribute {name}")
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
