import functools
import numbers

import numpy as np
from numpy.lib.arraysetops import isin
import scipy.sparse as ssp
import sparse


def _wrap_ret():
    """Wrap return value of func to spartan tensor types.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if isinstance(ret, np.ndarray):
                return DTensor(ret)
            elif isinstance(ret, sparse.SparseArray):
                t = STensor.from_sparse_array(ret)
                return STensor.from_sparse_array(ret)
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


class DTensor(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value):
        if isinstance(value, STensor):
            self._data = value._data.toarray()
        else:
            self._data = np.asarray(value)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (DTensor,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._data if isinstance(x, DTensor) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x._data if isinstance(x, DTensor) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

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
        return getattr(self._data, name)

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

    @classmethod
    def from_numpy(cls, x):
        t = cls.__new__(cls)
        t._data = np.asarray(x)
        return t


class STensor(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data, shape=None):
        if type(data) is tuple:
            indices, values = data
            self._data = sparse.COO(indices, values, shape=shape)
        else:
            self._data = sparse.as_coo(indices, values, shape=shape)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (STensor,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._data if isinstance(x, STensor) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x._data if isinstance(x, STensor) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            if isinstance(result, np.ndarray):
                return self.__class__.from_numpy(result)
            elif isinstance(result, ssp.spmatrix):
                return self.__class__.from_scipy_sparse(ret)
            elif isinstance(result, sparse.COO):
                return self.__class__.from_sparse_array(result)
            else:
                return result

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._data)

    @_wrap_ret()
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        return getattr(self._data, name)

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

    def __len__(self):
        return self._data.__len__()

    @_wrap_ret()
    def __getitem__(self, index):
        return self._data.__getitem__(index)

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def __delitem__(self, index):
        self._data.__delitem__(index)

    @_wrap_ret()
    @_check_params(1)
    def dot(self, other):
        return self._data.dot(other._data)

    @_wrap_ret()
    def todense(self):
        return self._data.todense()

    @classmethod
    def from_numpy(cls, x):
        t = cls.__new__(cls)
        t._data = sparse.COO.from_numpy(x)
        return t

    @classmethod
    def from_scipy_sparse(cls, x):
        t = cls.__new__(cls)
        t._data = sparse.COO.from_scipy_sparse(x)
        return t

    @classmethod
    def from_sparse_array(cls, x):
        if not isinstance(x, sparse.SparseArray):
            raise TypeError(
                f"Argument type should be `sparse.SparseArray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = x
        return t
