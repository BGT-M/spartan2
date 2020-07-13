import functools

import torch
import torch.sparse as sparse


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
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
