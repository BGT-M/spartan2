import functools
import warnings

import numpy as np
import sparse
import scipy
import scipy.sparse as ssp
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
    """A dense multi-deimensional tensor on GPU (based on PyTorch).

    Parameters
    ----------
    value : np.ndarray or array_like
        Data of dense tensor.
    dtype : data-type, optional
        Data type of the tensor, by default None(inferred automatically)

    Examples
    --------
    You can create a `DTensor` in many ways. For example, `numpy.ndarray`
    `torch.Tensor` and `list of list`.

    >>> x = np.random.rand(3, 4)
    >>> A = st.DTensor.from_numpy(x)
    >>> A
    DTensor(
    tensor([[0.9604, 0.9878, 0.4680, 0.0225],
            [0.5677, 0.3412, 0.6834, 0.1586],
            [0.6883, 0.5613, 0.8819, 0.2513]], device='cuda:0',
        dtype=torch.float64)
    )
    >>> A = st.DTensor(x)
    >>> A
    DTensor(
    tensor([[0.9604, 0.9878, 0.4680, 0.0225],
            [0.5677, 0.3412, 0.6834, 0.1586],
            [0.6883, 0.5613, 0.8819, 0.2513]], device='cuda:0',
        dtype=torch.float64)
    )
    >>> x = torch.rand(3, 4)
    >>> A = st.DTensor(x)
    >>> A
    DTensor(
    tensor([[0.8155, 0.3988, 0.6346, 0.5665],
            [0.2955, 0.8131, 0.0774, 0.7335],
            [0.0919, 0.8163, 0.5959, 0.9401]], device='cuda:0')
    )
    >>> A = st.DTensor([[1, 2, 3, 4], [2, 1, 4, 3], [1, 3, 2, 4]])
    >>> A
    DTensor(
    tensor([[1, 2, 3, 4],
            [2, 1, 4, 3],
            [1, 3, 2, 4]], device='cuda:0')
    )
    """

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

    def __copy__(self):
        return self._data.__copy__()

    def __iter__(self):
        return iter(self._data)

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
        """Wrapper of `torch.all`

        Parameters
        ----------
        axis : None or int or tuple of ints , optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or bool
            Result of `all` operation as a DTensor or a boolean value.
        """
        if axis is None:
            return torch.all(self._data)
        else:
            return torch.all(self._data, dim=axis, keepdim=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        """Wrapper of `torch.any`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or bool
            Result of `any` operation as a DTensor or a boolean value.
        """
        if axis is None:
            return torch.any(self._data)
        return torch.any(self._data, dim=axis, keepdim=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.min`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `min` operation as a DTensor or a scalar value.
        """
        return self._data.min(dim=axis, keepdim=keepdims).values

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.max`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `max` operation as a DTensor or a scalar value.
        """
        return self._data.max(dim=axis, keepdim=keepdims).values

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.sum`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `sum` operation as a DTensor or a scalar value.
        """
        if axis is None:
            return torch.sum(self._data)
        return self._data.sum(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.prod`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `prod` operation as a DTensor or a scalar value.
        """
        if axis is None:
            return torch.prod(self._data)
        return self._data.prod(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.mean`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `mean` operation as a DTensor or a scalar value.
        """
        if axis is None:
            return torch.mean(self._data)
        return self._data.sum(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.var`, note that result is biased(divisor is `n`).

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are \
                left to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `var` operation as a DTensor or a scalar value.
        """
        if axis is None:
            return torch.var(self._data)
        return self._data.var(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        """Wrapper of `torch.Tensor.std`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        DTensor or scalar
            Result of `std` operation as a DTensor or a scalar value.
        """
        if axis is None:
            return torch.std(self._data)
        return self._data.std(dim=axis, keepdim=keepdims)

    @_wrap_ret
    def dot(self, other):
        """Wrapper of `torch.Tensor.dot`. Perform tensor dot operation with
        another tensor, including vector inner product, matrix multiplication
        and general tensor dot.

        Parameters
        ----------
        other : DTensor
            The second operand of dot operation.

        Returns
        -------
        DTensor or scalar
            Tensor dot result as a DTensor or a scalar value.
        """
        if other.ndim == 1:
            return self._data.dot(other._data)
        else:
            return self._data.matmul(other._data)

    @_wrap_ret
    def reshape(self, shape):
        """Wrapper of `torch.Tensor.reshape`. Gives a new dense tensor with
        new shape and current data.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of new tensor.

        Returns
        -------
        DTensor
            New dense tensor with given shape.
        """
        return self._data.reshape(shape)

    @_wrap_ret
    def nonzero(self):
        """Wrapper of `torch.Tensor.nonzero`. Find the indices of non-zero elements.

        Returns
        -------
        tuple of DTensor
            Indices of non-zero elements.
        """
        return tuple([DTensor(d) for d in self._data.nonzero()])

    @_wrap_ret
    def astype(self, dtype):
        """Wrapper of `torch.Tensor.astype`. Convert the array to a new type.

        Parameters
        ----------
        dtype : data-type
            Type of new tensor.

        Returns
        -------
        DTensor
            New dense tensor with given data type.
        """
        return self._data.type(dtype)

    def to_numpy(self):
        """Return the tensor as `numpy.ndarray`.

        Returns
        -------
        numpy.ndarray
            Returned numpy array.
        """
        return self._data.cpu().numpy()

    @classmethod
    def from_numpy(cls, x):
        """Construct a `DTensor` from a `numpy.ndarray`.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        DTensor
            Dense tensor constructed from given array.

        Raises
        ------
        TypeError
            Raise if the parameter is not `numpy.ndarray`.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Argument type should be `numpy.ndarray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = torch.from_numpy(x).cuda()
        return t


class STensor(TorchArithmetic):
    """A sparse multi-dimensional tensor on GPU (based on PyTorch).

    Parameters
    ----------
    data : Tuple of indices (ndim x nnz) and values (1 x nnz) or STenosr
        or DTensor
        Data of sparse tensor.
    shape : tuple, optional
        Shape of sparse tensor, by default None (inferred automatically)
    """

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
        self._data = self._data.cuda().coalesce()

    def __repr__(self):
        return '%s(\n%r\n)' % (type(self).__name__, self._data)

    def __len__(self):
        return self._data.__len__()

    def __copy__(self):
        return self._data.__copy__()

    def __iter__(self):
        return iter(self._data)

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
        """Wrapper of `torch.all`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or bool
            Result of `all` operation as a STensor or a boolean value.
        """
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return torch.all(self._data)
        return torch.all(self._data, axis=axis, keepdim=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        """Wrapper of `torch.any`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or bool
            Result of `any` operation as a STensor or a boolean value.
        """
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return torch.any(self._data)
        return torch.any(self._data, axis=axis, keepdim=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        """Not implmented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `min` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        """Not implemented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `max` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        """Wrapper of `torch.sparse.sum`

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `sum` operation as a STensor or a scalar value.
        """
        if keepdims:
            warnings.warn(f"The keepdims parameter has no effect yet",
                          UserWarning)
        if axis is None:
            return tsparse.sum(self._data)
        return tsparse.sum(self._data, dim=axis)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        """Not implemented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `prod` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        """Not implemented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `mean` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        """Not implemented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `var` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        """Not implemented yet.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes to operate on, by default
            None
        keepdims : bool, optional
            If true, the axes along which the operation performed are left
            to size one, by default False

        Returns
        -------
        STensor or scalar
            Result of `std` operation as a STensor or a scalar value.
        """
        raise NotImplementedError

    @_wrap_ret
    def dot(self, other):
        """Wrapper of `torch.sparse.mm`. Perform tensor dot operation with
        another tensor, including vector inner product, matrix multiplication
        and general tensor dot.

        Parameters
        ----------
        other : DTensor or STensor
            The second operand of dot operation.

        Returns
        -------
        DTensor or STensor
            Tensor dot result as a DTensor or a STensor or a scalar value.
        """
        return tsparse.mm(self._data, other._data)

    @_wrap_ret
    def todense(self):
        """Convert to a dense tensor.

        Returns
        -------
        DTensor
            Dense tensor coverted from the sparse tensor.
        """
        return self._data.to_dense()

    @_wrap_ret
    def reshape(self, shape):
        """Not implemented yet.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of new tensor.

        Returns
        -------
        STensor
            New sparse tensor with given shape.
        """
        raise NotImplementedError

    @_wrap_ret
    def nonzero(self):
        """Wrapper of `torch.Tensor.indices`. Find the indices of non-zero elements.

        Returns
        -------
        tuple of DTensor
            Indices of non-zero elements.
        """
        return tuple([x for x in self._data.indices()])

    @_wrap_ret
    def astype(self, dtype):
        """Not implemented yet.

        Parameters
        ----------
        dtype : data-type
            Type of new tensor.

        Returns
        -------
        STensor
            New sparse tensor with given data type.
        """
        raise NotImplementedError

    def to_scipy(self, format='coo'):
        """Return the sparse tensor as a `scipy.sparse.spmatrix`.

        Parameters
        ----------
        format : str, optional, {'coo', 'csr', 'csc', 'lil', 'dok'}
            Format of scipy sparse matrix, by default 'coo'

        Returns
        -------
        scipy.sparse.spmatrix
            Returned scipy sparse matrix.
        """
        if self._data.ndim != 2:
            raise ValueError(
                f"Only 2-dim sparse tensor can be converted to scipy sparse\
                matrix, got {self._data.ndim}")
        indices = self._data.indices().cpu().numpy()
        row, col = indices[0], indices[1]
        vals = self._data.values().cpu().numpy()
        shape = tuple(self._data.shape)
        return ssp.coo_matrix((vals, (row, col)), shape=shape).asformat(format)

    def to_sparse_array(self):
        """Return the sparse tensor as a `sparse.COO`.

        Returns
        -------
        sparse.COO
            Returned sparse array.
        """
        indices = self._data.indices().cpu().numpy()
        vals = self._data.values().cpu().numpy()
        shape = tuple(self._data.shape)
        return sparse.COO(indices, vals, shape=shape)

    @classmethod
    def from_numpy(cls, x: np.ndarray):
        """Construct a `STensor` from a `numpy.ndarray`.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        STensor
            Sparse tensor constructed from given array.

        Raises
        ------
        TypeError
            Raise if the parameter is not `numpy.ndarray`.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Argument type should be `numpy.ndarray`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = torch.from_numpy(x).to_sparse().cuda()
        t._data = t._data.coalesce()
        return t

    @classmethod
    def from_scipy_sparse(cls, x: scipy.sparse.spmatrix):
        """Construct a `STensor` from a `scipy.sparse.spmatrix`.

        Parameters
        ----------
        x : scipy.sparse.spmatrix
            Input sparse matrix.

        Returns
        -------
        STensor
            Sparse tensor constructed from given sparse matrix.

        Raises
        ------
        TypeError
            Raise if the parameter is not `scipy.sparse.spmatrix`.
        """
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
        t._data = t._data.coalesce()
        return t

    @classmethod
    def from_sparse_array(cls, x: sparse.COO):
        """Construct a sparse tensor from a `sparse.COO`.

        Parameters
        ----------
        x : sparse.COO
            Input COO sparse array.

        Returns
        -------
        STensor
            Sparse tensor constructed from given sparse array.

        Raises
        ------
        TypeError
            Raise if the parameter is not `sparse.COO`.
        """
        if not isinstance(x, sparse.COO):
            raise TypeError(
                f"Argument type should be `sparse.SparseArray`, got {type(x)}")
        indices = torch.from_numpy(x.coords).type(torch.long)
        values = torch.from_numpy(x.data)
        shape = torch.Size(x.shape, dtype=torch.long)
        t = cls.__new__(cls)
        t._data = tsparse.FloatTensor(indices, values, shape).cuda()
        t._data = t._data.coalesce()
        return t

    @classmethod
    def from_torch(cls, x: torch.Tensor):
        """Construct a `STensor` from a `torch.Tensor`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        STensor
            Sparse tensor constructed from given array.

        Raises
        ------
        TypeError
            Raise if the parameter is not `torch.Tensor`.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Argument type should be `torch.Tensor`, got {type(x)}")
        t = cls.__new__(cls)
        if not x.is_sparse:
            x = x.to_sparse()
        t._data = x.cuda()
        t._data = t._data.coalesce()
        return t
