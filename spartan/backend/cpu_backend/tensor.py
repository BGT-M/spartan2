import functools
import numbers

import numpy as np
import scipy.sparse as ssp
import sparse


def _ensure_array(x):
    if isinstance(x, DTensor):
        return x._data
    return np.asarray(x)


def _ensure_tensor(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return DTensor(x)
    elif isinstance(x, sparse.SparseArray):
        return STensor.from_sparse_array(x)
    else:
        return x


def _wrap_ret(func):
    """Wrap return value of func to spartan tensor types.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if type(ret) is tuple:
            return tuple(_ensure_tensor(x) for x in ret)
        else:
            return _ensure_tensor(ret)
    return wrapper


class DTensor(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value, dtype=None):
        if isinstance(value, STensor):
            self._data = value._data.todense().astype(dtype)
        elif isinstance(value, DTensor):
            self._data = value._data.astype(dtype)
        else:
            self._data = np.asarray(value, dtype=dtype)

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

    def __copy__(self):
        return self._data.__copy__()

    # Slice
    @_wrap_ret
    def __getitem__(self, index):
        return self._data.__getitem__(index)

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def __delitem__(self, index):
        self._data.__delitem__(index)

    @_wrap_ret
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

    @_wrap_ret
    def all(self, axis=None, keepdims=False):
        return np.all(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        return np.any(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        return self._data.min(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        return self._data.max(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        return self._data.prod(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        return self._data.var(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        return self._data.std(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def dot(self, other):
        return self._data.dot(other._data)

    @_wrap_ret
    def reshape(self, shape):
        return self._data.reshape(shape)

    @_wrap_ret
    def nonzero(self):
        return tuple([DTensor(d) for d in self._data.nonzero()])

    @_wrap_ret
    def astype(self, dtype):
        return self._data.astype(dtype)

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Argument type should be `numpy.ndarray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = np.asarray(x)
        return t


class STensor(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data, shape=None):
        if type(data) is tuple:
            indices, values = data
            self._data = sparse.COO(indices, values, shape=shape)
        elif isinstance(data, DTensor):
            self._data = sparse.as_coo(data._data, shape=shape)
        elif isinstance(data, STensor):
            self._data = data._data
        else:
            self._data = sparse.as_coo(data, shape=shape)

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
                return self.__class__.from_scipy_sparse(result)
            elif isinstance(result, sparse.COO):
                return self.__class__.from_sparse_array(result)
            else:
                return result

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._data)

    def __copy__(self):
        return self._data.__copy__()

    @_wrap_ret
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

    def __len__(self):
        return self._data.__len__()

    @_wrap_ret
    def __getitem__(self, index):
        return self._data.__getitem__(index)

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def __delitem__(self, index):
        self._data.__delitem__(index)

    @_wrap_ret
    def all(self, axis=None, keepdims=False):
        return np.all(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        return np.any(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        return self._data.min(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        return self._data.max(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        return self._data.prod(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        return self._data.var(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        return self._data.std(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def dot(self, other):
        return self._data.dot(other._data)

    @_wrap_ret
    def todense(self):
        return self._data.todense()

    @_wrap_ret
    def reshape(self, shape):
        return self._data.reshape(shape)

    @_wrap_ret
    def nonzero(self):
        return self._data.nonzero()

    @_wrap_ret
    def astype(self, dtype):
        return self._data.astype(dtype)

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Argument type should be `numpy.ndarray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = sparse.COO.from_numpy(x)
        return t

    @classmethod
    def from_scipy_sparse(cls, x: ssp.spmatrix):
        if not isinstance(x, ssp.spmatrix):
            raise TypeError(
                f"Argument type should be `scipy.sparse.spmatrix`, \
                got {type(x)}")
        t = cls.__new__(cls)
        t._data = sparse.COO.from_scipy_sparse(x)
        return t

    @classmethod
    def from_sparse_array(cls, x: sparse.COO):
        if not isinstance(x, sparse.COO):
            raise TypeError(
                f"Argument type should be `sparse.COO`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = x
        return t
