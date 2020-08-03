import builtins
import functools
import math

import torch

from .tensor import DTensor, STensor, _wrap_ret, _ensure_tensor


def _dispatch(dfunc, sfunc=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_sparse = builtins.any([isinstance(x, STensor) for x in args])
            is_dense = builtins.any([isinstance(x, DTensor) for x in args])
            if (is_sparse and is_dense):
                raise TypeError(
                    f"Parameters of `st.{func.__name__}` should be all STensor \
                    or all DTensor.")
            if is_sparse:
                if sfunc is None:
                    raise TypeError(f"`st.{func.__name__}` doesn't support \
                                    STensor parameters.")
                args = tuple(x._data if isinstance(x, STensor) else x
                             for x in args)
                return _ensure_tensor(sfunc(*args, **kwargs))
            else:
                args = tuple(x._data if isinstance(x, DTensor) else x
                             for x in args)
                return _ensure_tensor(dfunc(*args, **kwargs))
        return wrapper
    return decorator


# Type definition
short = torch.short
uint8 = torch.uint8
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
complex64 = torch.complex64
complex128 = torch.complex128


@_wrap_ret
def add(input_, other):
    """Wrapper of `torch.add`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The first operand.
    other : DTensor or STensor
        The second operand.

    Returns
    -------
    DTensor or STensor
        Output tensor.
    """
    return torch.add(input_._data, other._data)


@_wrap_ret
def all(input_, axis=None, keepdims=False):
    """Wrapper of `torch.all`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False

    Returns
    -------
    DTensor or bool
        Output tensor.
    """
    if axis is None:
        return torch.all(input_._data)
    return torch.all(input_._data, dim=axis, keepdim=keepdims)


@_wrap_ret
def angle(input_, deg=False):
    """Wrapper of `torch.angle`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    deg : bool, optional
        If true, result is in degree format. Otherwise, return in radians. By
        default False
    """
    if deg:
        ret = torch.angle(input_) * 180 / math.pi
    else:
        ret = torch.angle(input_)
    return ret


@_wrap_ret
def any(input_, axis=None, keepdims=False):
    """Wrapper of `torch.any`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False

    Returns
    -------
    DTensor or bool:
    """
    if axis is None:
        return torch.any(input_._data)
    return torch.any(input_._data, dim=axis, keepdim=keepdims)


@_wrap_ret
def arange(start, stop, step, dtype=None):
    """Wrapper of `torch.arange`

    Parameters
    ----------
    start : number
        Start of the interval.
    stop : number
        End of the interval.
    step : number
        Spacing between spaces.
    dtype : data-type, optional
        Type of the return tensor, by default None

    Returns
    -------
    DTensor
        Constructed dense tensor.
    """
    return torch.arange(start, stop, step, dtype=dtype)


@_wrap_ret
def argmax(input_, axis=None):
    """Wrapper of `torch.argmax`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    if axis is None:
        return torch.argmax(input_)
    else:
        return torch.argmax(input_, dim=axis)


@_wrap_ret
def argmin(input_, axis=None):
    """Wrapper of `torch.argmin`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    if axis is None:
        return torch.argmin(input_)
    else:
        return torch.argmin(input_, dim=axis)


@_wrap_ret
def argsort(input_, axis=-1):
    """Wrapper of `torch.argsort`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    return torch.argsort(input_._data, dim=axis)


@_wrap_ret
def bincount(input_, weights=None, minlength=0):
    """Wrapper of `torch.bincount`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    weights : DTensor
        Weights.
    minlength : int, optional
        Minimum number of bins, by default 0
    """
    return torch.bincount(input_._data, weights, minlength)


@_wrap_ret
def bitwise_and(input_, other):
    """Wrapper of `torch.bitwise_and`

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.bitwise_and(input_._data, other._data)


@_wrap_ret
def bitwise_not(input_):
    """Wrapper of `torch.bitwise_not`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.bitwise_not(input_._data)


@_wrap_ret
def bitwise_or(input_, other):
    """Wrapper of `torch.bitwise_or`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.bitwise_or(input_._data, other._data)


@_wrap_ret
def bitwise_xor(input_, other):
    """Wrapper of `torch.bitwise_xor`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.bitwise_xor(input_._data, other._data)


def can_cast(from_, to):
    """Wrapper of `torch.can_cast`.

    Parameters
    ----------
    from_ : data-type
        Data type to cast from.
    to : data-type
        Data type to cast to.
    """
    return torch.can_cast(from_, to)


@_wrap_ret
def ceil(input_):
    """Wrapper of `torch.ceil`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.ceil(input_._data)


@_wrap_ret
def concatenate(inputs, axis=None):
    """Wrapper of `torch.cat`

    Parameters
    ----------
    inputs : sequence of DTensor
        Dense tensor to be concatenated.
    axis : int or None, optional
        Axis to operate on, by default None
    """
    return torch.cat((x._data for x in inputs), dim=axis)


@_wrap_ret
def conj(input_):
    """Wrapper of `torch.conj`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.conj(input_._data)


@_wrap_ret
def cos(input_):
    """Wrapper of `torch.cos`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.cos(input_._data)


@_wrap_ret
def cosh(input_):
    """Wrapper of `torch.cosh`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.cos(input_._data)


@_wrap_ret
def cumprod(input_, axis=None, dtype=None):
    """Wrapper of `torch.cumprod`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int, optional
        Axis to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    """
    return torch.cumprod(input_._data, dim=axis, dtype=dtype)


@_wrap_ret
def cumsum(input_, axis=None, dtype=None):
    """Wrapper of `torch.cumsum`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int, optional
        Axis to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    """
    return torch.cumsum(input_._data, axis, dtype)


@_wrap_ret
def diag(input_, k=0):
    """Wrapper of `torch.diag`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    k : int, optional
        Offset to main diagonal, by default 0
    """
    return torch.diag(input_._data, diagonal=k)


@_wrap_ret
def diagflat(input_, k=0):
    """Wrapper of `torch.diagflat`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    return torch.diagflat(input_._data, k)


@_wrap_ret
def diagonal(input_, offset=0, axis1=0, axis2=1):
    """Wrapper of `torch.diagonal`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    offset : int, optional
        Offset to the main diagonal, by default 0
    axis1 : int, optional
        The first axis of the diagonal, by default 0
    axis2 : int, optional
        The second axis of the diagonal, by default 1
    """
    return torch.diagonal(input_._data, offset=offset, dim1=axis1, dim2=axis2)


@_wrap_ret
def dot(input_, other):
    """Wrapper of `torch.dot`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    if input_._data.ndim == 1 and other._data.ndim == 1:
        return torch.dot(input_._data, other._data)
    return torch.matmul(input_._data, other._data)


@_wrap_ret
def empty(shape, dtype):
    """Wrapper of `torch.empty`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of tensor.
    dtype : data-type
        Data type of tensor.
    """
    return torch.empty(shape, dtype=dtype)


@_wrap_ret
def empty_like(input_, dtype):
    """Wrapper of `torch.empty_like`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    dtype : data-type
        Data type of output.
    """
    return torch.empty_like(input_._data, dtype=dtype)


@_wrap_ret
def equal(input_, other):
    """Wrapper of `torch.equal`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.equal(input_._data, other._data)


@_wrap_ret
def exp(input_):
    """Wrapper of `torch.exp`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.exp(input_._data)


@_wrap_ret
def expm1(input_):
    """Wrapper of `torch.expm1`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    return torch.expm1(input_._data)


@_wrap_ret
def eye(n, m=None, dtype=None):
    """Wrapper of `torch.eye`.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of cols, by default None (equal to `n`)
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.eye(n, m, dtype=dtype)


@_wrap_ret
def flip(input_, axis=None):
    """Wrapper of `torch.flip`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    if axis is None:
        axis = list(range(input_.data.ndim))
    return torch.flip(input_._data, axis)


@_wrap_ret
def floor(input_):
    """Wrapper of `torch.floor`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.floor(input_._data)


@_wrap_ret
def floor_divide(input_, other):
    """Wrapper of `torch.floor_divide`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.floor_divide(input_._data, other._data)


@_wrap_ret
def fmod(input_, other):
    """Wrapper of `torch.fmod`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.fmod(input_._data, other._data)


@_wrap_ret
def full(shape, value, dtype=None):
    """Wrapper of `torch.full`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    value : scalar
        Fill value of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.full(shape, value, dtype=dtype)


@_wrap_ret
def full_like(input_, value, dtype=None):
    """Wrapper of `numpy.full_like`.

    Parameters
    ----------
    input_ : DTensor
        The input tensor.
    value : scalar
        Fill value of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.full_like(input_._data, value, dtype=dtype)


@_wrap_ret
def imag(input_):
    """Wrapper of `torch.imag`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.imag(input_._data)


@_wrap_ret
def isfinite(input_):
    """Wrapper of `torch.isinfinite`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.isfinite(input_._data)


@_wrap_ret
def isinf(input_):
    """Wrapper of `torch.isinf`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.isinf(input_._data)


@_wrap_ret
def isnan(input_):
    """Wrapper of `torch.isnan`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.isnan(input_._data)


@_wrap_ret
def linspace(start, end, step, dtype=None):
    """Wrapper of `torch.linspace`.

    Parameters
    ----------
    start : DTensor
        Dense tensor of start points.
    end : DTensor
        Dense tensor of end points.
    step : int
        Number of sequence.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.linspace(start, end, step, dtype=dtype)


@_wrap_ret
def log(input_):
    """Wrapper of `torch.log`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.log(input_._data)


@_wrap_ret
def log10(input_):
    """Wrapper of `torch.log10`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.log10(input_._data)


@_wrap_ret
def log1p(input_):
    """Wrapper of `torch.log1p`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.log1p(input_._data)


@_wrap_ret
def log2(input_):
    """Wrapper of `torch.log2`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.log2(input_._data)


@_wrap_ret
def logical_and(input_, other):
    """Wrapper of `torch.logical_and`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.logical_and(input_._data, other._data)


@_wrap_ret
def logical_not(input_):
    """Wrapper of `torch.logical_not`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.logical_not(input_._data)


@_wrap_ret
def logical_or(input_, other):
    """Wrapper of `torch.logical_or`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.logical_or(input_._data, other._data)


@_wrap_ret
def logical_xor(input_, other):
    """Wrapper of `torch.logical_xor`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.logical_xor(input_._data, other._data)


@_wrap_ret
def logspace(start, stop, step, base=10, dtype=None):
    """Wrapper of `torch.logspace`.

    Parameters
    ----------
    start : DTensor
        Dense tensor of start points.
    stop : DTensor
        Dense tensor of end points.
    step : int
        Number of sequence.
    base : int, optional
        Base of the logarithm, by default 10
    dtype : data-type, optional
        Data type of output, by default None
    """
    return torch.logspace(start, stop, step, base=base, dtype=dtype)


@_wrap_ret
def matmul(input_, other):
    """Wrapper of `torch.matmul`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The first operand.
    other : DTensor or STensor
        The second operand.
    """
    return torch.matmul(input_._data, other._data)


@_wrap_ret
def mean(input_, axis=None, keepdims=False):
    """Wrapper of `torch.mean`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    if axis is None:
        ret = torch.mean(input_._data)
    else:
        ret = torch.mean(input_._data, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def median(input_, axis=-1, keepdims=False):
    """Wrapper of `torch.median`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    if axis is None:
        ret = torch.median(input_._data)
    else:
        ret = torch.median(input_._data, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def meshgrid(*inputs):
    """Wrapper of `numpy.meshigrid`.
    """
    datas = [i._data for i in inputs]
    return tuple([d for d in torch.meshgrid(*datas)])


@_wrap_ret
def nonzero(input_):
    """Wrapper of `torch.nonzero`

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return tuple([d for d in torch.nonzero(input_._data, as_tuple=True)])


@_wrap_ret
def ones(shape, dtype=None):
    """Wrapper of `torch.ones`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.ones(shape, dtype=dtype)


@_wrap_ret
def ones_like(input_, dtype=None):
    """Wrapper of `torch.ones_like`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.ones_like(input_._data, dtype=dtype)


@_wrap_ret
def prod(input_, axis=None, keepdims=False, dtype=None):
    """Wrapper of `torch.prod`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    dtype : data-type, optional
        Data type of output, by default None
    """
    if axis is None:
        return torch.prod(input_._data, dtype=dtype)
    return torch.prod(input_._data, dim=axis, keepdim=keepdims, dtype=dtype)


@_wrap_ret
def real(input_):
    """Wrapper of `torch.real`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.real(input_._data)


@_wrap_ret
def reciprocal(input_):
    """Wrapper of `torch.reciprocal`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.reciprocal(input_._data)


@_wrap_ret
def remainder(input_, other):
    """Wrapper of `torch.remainder`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.remainder(input_._data, other._data)


@_wrap_ret
def reshape(input_, shape):
    """Wrapper of `numpy.rehsape`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    shape : tuple of ints
        Shape of new tensor.
    """
    return torch.rehsape(input_._data, shape)


@_wrap_ret
def roll(input_, shift, axis=None):
    """Wrapper of `torch.roll`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    shift : int or tuple of ints
        Shift numbers of each axis or axes.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    return torch.roll(input_._data, shift, dims=axis)


@_wrap_ret
def rot90(input_, k=1, axes=(0, 1)):
    """Wrapper of `torch.rot90`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    k : int, optional
        Number of rotation times, by default 1
    axes : tuple, optional
        The axes in which the input is rotated, by default (0, 1)
    """
    return torch.rot90(input_._data, k, dims=axes)


@_wrap_ret
def sign(input_):
    """Wrapper of `torch.sign`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.sign(input_._data)


@_wrap_ret
def sin(input_):
    """Wrapper of `torch.sin`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.sin(input_._data)


@_wrap_ret
def sinh(input_):
    """Wrapper of `torch.sinh`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.sinh(input_._data)


@_wrap_ret
def split(input_, indices_or_sections, axis=0):
    """Wrapper of `torch.split`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    indices_or_sections : int or DTensor
        If integer, specifing part of splits. Else specifing the positions
        where the splits take place.
    axis : int, optional
        The axis to operate on, by default 0
    """
    return torch.split(input_._data, indices_or_sections, axis)


@_wrap_ret
def sqrt(input_):
    """Wrapper of `torch.sqrt`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.sqrt(input_._data)


@_wrap_ret
def square(input_):
    """Wrapper of `torch.square`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.square(input_._data)


@_wrap_ret
def squeeze(input_, axis=None):
    """Wrapper of `torch.squeeze`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    return torch.squeeze(input_._data, dim=axis)


@_wrap_ret
def stack(inputs, axis=0):
    """Wrapper of `torch.stack`.

    Parameters
    ----------
    inputs : Sequence of DTensors.
        Dense tensors to be stacked, must have same shape.
    axis : int, optional
        Axis to operate on, by default 0
    """
    return torch.stack((x._data for x in inputs), dim=axis)


@_wrap_ret
def std(input_, axis=None, keepdims=False):
    """Wrapper of `torch.std`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    if axis is None:
        ret = torch.std(input_._data, unbiased=False)
    else:
        ret = torch.std(input_._data, dim=axis,
                        keepdim=keepdims, unbiased=False)
    return ret


@_wrap_ret
def sum(input_, axis=None, dtype=None, keepdims=False):
    """Wrapper of `numpy.sum`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    if axis is None:
        ret = torch.sum(input_._data, dtype=dtype)
    else:
        ret = torch.sum(input_._data, dim=axis, dtype=dtype, keepdim=keepdims)
    return ret


@_wrap_ret
def take(input_, indices):
    """Wrapper of `torch.take`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    indices : DTensor
        Indices of extracted values.
    """
    return torch.take(input_._data, indices)


@_wrap_ret
def tan(input_):
    """Wrapper of `torch.tan`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.tan(input_._data)


@_wrap_ret
def tanh(input_):
    """Wrapper of `torch.tanh`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.tanh(input_._data)


@_wrap_ret
def tensordot(input_, other, axes=2):
    """Wrapper of `numpy.tensordot`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor or STensor
        The second operand.
    axes : int or tuple of ints, optional
        Axis or axes along which the operation is performed, by default None
    """
    return torch.tensordot(input_._data, other._data, axes)


@_wrap_ret
def trace(input_):
    """Wrapper of `torch.trace`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    """
    return torch.trace(input_._data)


@_wrap_ret
def transpose(input_, axes=None):
    """Wrapper of `torch.transpose`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor
    axes : list of ints, optional
       Axes along which the operation is performed, by default None
    """
    if axes is None:
        axes = (0, 1)
    return torch.transpose(input_, axes[0], axes[1])


@_wrap_ret
def tril(input_, k=0):
    """Wrapper of `torch.tril`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    k : int, optional
        Offset to main diagonal, by default 0
    """
    return torch.tril(input_._data, k)


@_wrap_ret
def tril_indices(n, m=0, k=0):
    """Wrapper of `torch.tril_indices`.

    Parameters
    ----------
    n : int
        Number of row of output tensor.
    m : int, optional
        Number of column of output tensor, by default 0
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    return torch.tril_indices(row=m, col=m, offset=k)


def triu(input_, k=0):
    """Wrapper of `torch.triu`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor
    k : int, optional
        Offset to main diagonal, by default 0
    """
    return torch.triu(input_._data, k)


@_wrap_ret
def triu_indices(n, m=0, k=0):
    """Wrapper of `torch.triu_indices`.

    Parameters
    ----------
    n : int
        Number of row of output tensor.
    m : int, optional
        Number of column of output tensor, by default 0
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    ret = torch.triu_indices(row=m, col=m, offset=k)
    return tuple([index for index in ret])


@_wrap_ret
def true_divide(input_, other):
    """Wrapper of `torch.true_divide`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    return torch.true_divide(input_._data, other._data)


@_wrap_ret
def trunc(input_):
    """Wrapper of `torch.trunc`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    return torch.trunc(input_._data)


@_wrap_ret
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    """Wrapper of `torch.unique`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    return_inverse : bool, optional
        If True, return the indices also, by default False
    return_counts : bool, optional
        If True, return the count also, by default False
    axis : int or None, optional
        The axis to operate on, by default None
    """
    return torch.unique(input_._data, return_inverse=return_inverse,
                        return_counts=return_counts, dim=axis)


@_wrap_ret
def var(input_, axis=None, keepdims=False):
    """Wrapper of `torch.var`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    if axis is None:
        ret = torch.var(input_)
    else:
        ret = torch.var(input_, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def where(condition, x, y):
    """Wrapper of `torch.where`.

    Parameters
    ----------
    condition : DTensor of bool
        Where True, yield x, otherwise yield y.
    x : DTensor
        The first tensor.
    y : DTensor
        The second tensor.
    """
    return torch.where(condition, x, y)


@_wrap_ret
def zeros(shape, dtype=None):
    """Wrapper of `torch.zeros`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.zeros(shape, dtype=dtype)


@_wrap_ret
def zeros_like(input_, dtype=None):
    """Wrapper of `torch.zeros_like`.

    Parameters
    ----------
    input_ : DTensor
        Input tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    return torch.zeros_like(input_._data, dtype=dtype)
