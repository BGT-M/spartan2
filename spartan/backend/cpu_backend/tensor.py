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
    """A dense multi-deimensional tensor on CPU (based on NumPy).

    Parameters
    ----------
    value : np.ndarray or array_like
        Data of dense tensor.
    dtype : data-type, optional
        Data type of the tensor, by default None(inferred automatically)
    """

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
        """Wrapper of `numpy.all`

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
        return np.all(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        """Wrapper of `numpy.any`

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
        return np.any(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.min`

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
        return self._data.min(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.max`

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
        return self._data.max(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.sum`

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
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.prod`

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
        return self._data.prod(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.mean`

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
        return self._data.mean(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.var`, note that result is biased(divisor is `n`).

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
        return self._data.var(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        """Wrapper of `numpy.ndarray.std`

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
        return self._data.std(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def dot(self, other):
        """Wrapper of `numpy.ndarray.dot`. Perform tensor dot operation with
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
        return self._data.dot(other._data)

    @_wrap_ret
    def reshape(self, shape):
        """Wrapper of `numpy.ndarray.reshape`. Gives a new dense tensor with
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
        """Wrapper of `numpy.ndarray.nonzero`. Find the indices of non-zero elements.

        Returns
        -------
        tuple of DTensor
            Indices of non-zero elements.
        """
        return tuple([DTensor(d) for d in self._data.nonzero()])

    @_wrap_ret
    def astype(self, dtype):
        """Wrapper of `numpy.ndarray.astype`. Convert the array to a new type.

        Parameters
        ----------
        dtype : data-type
            Type of new tensor.

        Returns
        -------
        DTensor
            New dense tensor with given data type.
        """
        return self._data.astype(dtype)

    @classmethod
    def from_numpy(cls, x: np.ndarray):
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
        t._data = np.asarray(x)
        return t


class STensor(np.lib.mixins.NDArrayOperatorsMixin):
    """A sparse multi-dimensional tensor on CPU (based on sparse).

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.spmatrix or tuple of indices (ndim
    x nnz) and values (1 x nnz)
        Data of sparse tensor.
    shape : tuple, optional
        Shape of sparse tensor, by default None (inferred automatically)
    """

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
        """Wrapper of `numpy.all`

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
        return np.all(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def any(self, axis=None, keepdims=False):
        """Wrapper of `numpy.any`

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
        return np.any(self._data, axis=axis, keepdims=keepdims)

    @_wrap_ret
    def min(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.min`

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
        return self._data.min(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def max(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.max`

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
        return self._data.max(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def sum(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.sum`

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
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def prod(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.prod`

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
        return self._data.prod(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def mean(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.mean`

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
        return self._data.sum(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def var(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.var`

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
        return self._data.var(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def std(self, axis=None, keepdims=False):
        """Wrapper of `sparse.COO.std`

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
        return self._data.std(axis=axis, keepdims=keepdims)

    @_wrap_ret
    def dot(self, other):
        """Wrapper of `sparse.COO.dot`. Perform tensor dot operation with
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
        return self._data.dot(other._data)

    @_wrap_ret
    def todense(self):
        """Convert to a dense tensor.

        Returns
        -------
        DTensor
            Dense tensor coverted from the sparse tensor.
        """
        return self._data.todense()

    @_wrap_ret
    def reshape(self, shape):
        """Wrapper of `sparse.COO.reshape`. Gives a new dense tensor with
        new shape and current data.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of new tensor.

        Returns
        -------
        STensor
            New sparse tensor with given shape.
        """
        return self._data.reshape(shape)

    @_wrap_ret
    def nonzero(self):
        """Wrapper of `sparse.COO.nonzero`. Find the indices of non-zero elements.

        Returns
        -------
        tuple of DTensor
            Indices of non-zero elements.
        """
        return self._data.nonzero()

    @_wrap_ret
    def astype(self, dtype):
        """Wrapper of `sparse.COO.astype`. Convert the array to a new type.

        Parameters
        ----------
        dtype : data-type
            Type of new tensor.

        Returns
        -------
        STensor
            New sparse tensor with given data type.
        """
        return self._data.astype(dtype)

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
        t._data = sparse.COO.from_numpy(x)
        return t

    @classmethod
    def from_scipy_sparse(cls, x: ssp.spmatrix):
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
        if not isinstance(x, ssp.spmatrix):
            raise TypeError(
                f"Argument type should be `scipy.sparse.spmatrix`, \
                got {type(x)}")
        t = cls.__new__(cls)
        t._data = sparse.COO.from_scipy_sparse(x)
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
                f"Argument type should be `sparse.COO`, got {type(x)}")
        t = cls.__new__(cls)
        t._data = x
        return t
