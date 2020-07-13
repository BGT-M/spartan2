import functools
import warnings

import numpy as np
import sparse
import scipy
import torch
import torch.sparse as tsparse


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        if x.is_sparse:
            return STensor.from_torch(x)
        return DTensor(x)
    elif isinstance(x, torch.Size):
        return DTensor(torch.as_tensor(x))
    else:
        return x


def _wrap_ret(func):
    """Wrap return value of func to spartan tensor types.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        try:
            ret = torch.as_tensor(ret).cuda()
        except Exception:
            pass
        return _ensure_tensor(ret)
    return wrapper


class TorchArithmetic:
    # Unary Opeartors
    @_wrap_ret
    def __neg__(self):
        return -self._data

    @_wrap_ret
    def __pos__(self):
        return +self._data

    @_wrap_ret
    def __invert__(self):
        return ~self._data

    # Binary Operators
    @_wrap_ret
    def __add__(self, other):
        try:
            return self._data + other._data
        except AttributeError:
            return self._data + other

    @_wrap_ret
    def __mul__(self, other):
        try:
            return self._data * other._data
        except AttributeError:
            return self._data * other

    @_wrap_ret
    def __truediv__(self, other):
        try:
            return self._data / other._data
        except AttributeError:
            return self._data / other

    @_wrap_ret
    def __floordiv__(self, other):
        try:
            return self._data // other._data
        except AttributeError:
            return self._data // other

    @_wrap_ret
    def __mod__(self, other):
        try:
            return self._data % other._data
        except AttributeError:
            return self._data % other

    @_wrap_ret
    def __pow__(self, other, modulo=None):
        try:
            return self._data ** other._data
        except AttributeError:
            return self._data ** other

    @_wrap_ret
    def __matmul__(self, other):
        try:
            return self._data @ other._data
        except AttributeError:
            return self._data @ other

    @_wrap_ret
    def __and__(self, other):
        try:
            return self._data & other._data
        except AttributeError:
            return self._data & other

    @_wrap_ret
    def __or__(self, other):
        try:
            return self._data | other._data
        except AttributeError:
            return self._data | other

    @_wrap_ret
    def xor(self, other):
        try:
            return self._data ^ other._data
        except AttributeError:
            return self._data ^ other

    @_wrap_ret
    def __lshift__(self, other):
        return self._data << other

    @_wrap_ret
    def __rshift__(self, other):
        return self._data >> other

    # Comparison Operators
    @_wrap_ret
    def __eq__(self, other):
        try:
            return self._data == other._data
        except AttributeError:
            return self._data == other

    @_wrap_ret
    def __gt__(self, other):
        try:
            return self._data > other._data
        except AttributeError:
            return self._data > other

    @_wrap_ret
    def __lt__(self, other):
        try:
            return self._data < other._data
        except AttributeError:
            return self._data < other

    @_wrap_ret
    def __ge__(self, other):
        try:
            return self._data >= other._data
        except AttributeError:
            return self._data >= other

    @_wrap_ret
    def __le__(self, other):
        try:
            return self._data <= other._data
        except AttributeError:
            return self._data <= other


class DTensor(TorchArithmetic):
    def __init__(self, data, dtype=None):
        if isinstance(data, DTensor):
            self._data = data._data
        elif isinstance(data, STensor):
            self._data = data._data.to_dense()
        else:
            self._data = torch.as_tensor(data, dtype=dtype)
        self._data: torch.Tensor = self._data.cuda()

    def __repr__(self):
        return '%s(\n%r\n)' % (type(self).__name__, self._data)

    def __len__(self):
        return self._data.__len__()

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
        if axis is None:
            return torch.all(self._data)
        else:
            return torch.all(self._data, dim=axis, keepdim=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        if axis is None:
            return torch.any(self._data)
        return torch.any(self._data, dim=axis, keepdim=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        return self._data.min(dim=axis, keepdim=keepdims).values

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        return self._data.max(dim=axis, keepdim=keepdims).values

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(self._data)
        return self._data.sum(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        if axis is None:
            return torch.prod(self._data)
        return self._data.prod(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return torch.mean(self._data)
        return self._data.sum(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        if axis is None:
            return torch.var(self._data)
        return self._data.var(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        if axis is None:
            return torch.std(self._data)
        return self._data.std(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def dot(self, other):
        if other.ndim == 1:
            return self._data.dot(other._data)
        else:
            return self._data.matmul(other._data)

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
    def from_numpy(cls, x):
        t = cls.__new__(cls)
        t._data = torch.as_tensor(x).cuda()
        return t

    @classmethod
    def from_torch(cls, x):
        t = cls.__new__(cls)
        t._data = torch.as_tensor(x).cuda()
        return t


class STensor(TorchArithmetic):
    def __init__(self, data, shape=None):
        if type(data) is tuple:
            indices, values = data
            self._data = tsparse.FloatTensor(indices, values, shape=shape)
        elif isinstance(data, DTensor):
            self._data = data._data.to_sparse()
        elif isinstance(data, STensor):
            self._data = data._data
        else:
            self._data = tsparse.FloatTensor(data, shape=shape)
        self._data = self._data.cuda()

    def __repr__(self):
        return '%s(\n%r\n)' % (type(self).__name__, self._data)

    def __len__(self):
        return self._data.__len__()

    # Notice that Pytorch's GPU sparse tensor doesn't support slice fully yet!
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
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return torch.all(self._data)
        return torch.all(self._data, axis=axis, keepdim=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return torch.any(self._data)
        return torch.any(self._data, axis=axis, keepdim=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return tsparse.sum(self._data)
        return tsparse.sum(self._data, dim=axis)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        raise NotImplementedError

    @_wrap_ret
    def dot(self, other):
        return tsparse.mm(self._data, other._data)

    @_wrap_ret
    def todense(self):
        return self._data.to_dense()

    @_wrap_ret
    def reshape(self, shape):
        raise NotImplementedError

    @_wrap_ret
    def nonzero(self):
        return tuple([x for x in self._data.indices()])

    @_wrap_ret
    def astype(self, dtype):
        raise NotImplementedError

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Argument type should be `numpy.ndarray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = torch.from_numpy(x).to_sparse().cuda()
        return t

    @classmethod
    def from_scipy_sparse(cls, x: scipy.sparse.spmatrix):
        if not isinstance(x, scipy.sparse.spmatrix):
            raise TypeError(
                f"Argument type should be `scipy.sparse.spmatrix`, \
                got {type(x)}")
        x = x.tocoo()
        indices = torch.from_numpy(np.vstack([x.row, x.col])).type(torch.long)
        values = torch.from_numpy(x.data)
        shape = torch.Size(x.shape, dtype=torch.long)
        t = cls.__new__(cls)
        t._data = tsparse.FloatTensor(indices, values, shape).cuda()
        return t

    @classmethod
    def from_sparse_array(cls, x: sparse.COO):
        if not isinstance(x, sparse.COO):
            raise TypeError(
                f"Argument type should be `sparse.SparseArray`, got {type(x)}")
        indices = torch.from_numpy(x.coords).type(torch.long)
        values = torch.from_numpy(x.data)
        shape = torch.Size(x.shape, dtype=torch.long)
        t = cls.__new__(cls)
        t._data = tsparse.FloatTensor(indices, values, shape).cuda()
        return t
