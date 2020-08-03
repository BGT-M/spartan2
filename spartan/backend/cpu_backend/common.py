import builtins
import functools

import numpy as np
import sparse

from .tensor import DTensor, STensor, _ensure_tensor, _wrap_ret


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
short = np.short
uint8 = np.uint8
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128


@_wrap_ret
@_dispatch(np.add, np.add)
def add(input_, other):
    """Wrapper of `numpy.add`.

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
    pass


@_wrap_ret
@_dispatch(np.all, sparse.COO.all)
def all(input_, axis=None, keepdims=False):
    """Wrapper of `numpy.all` and `sparse.COO.all`

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False

    Returns
    -------
    DTensor or STensor or bool
        Output tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.angle)
def angle(input_, deg=False):
    """Wrapper of `numpy.angle`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    deg : bool, optional
        If true, result is in degree format. Otherwise, return in radians. By
        default False
    """
    pass


@_wrap_ret
@_dispatch(np.any, sparse.COO.any)
def any(input_, axis=None, keepdims=False):
    """Wrapper of `numpy.any` and `sparse.COO.any`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False

    Returns
    -------
    DTensor or STensor or bool:
    """
    pass


@_wrap_ret
def arange(start, stop, step, dtype=None):
    """Wrapper of `numpy.arange`

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
    return np.arange(start, stop, step, dtype=dtype)


@_wrap_ret
@_dispatch(np.argmax)
def argmax(input_, axis=None):
    """Wrapper of `numpy.argmax`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.argmin)
def argmin(input_, axis=None):
    """Wrapper of `numpy.argmin`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.argsort)
def argsort(input_, axis=-1):
    """Wrapper of `numpy.argsort`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.bincount)
def bincount(input_, weights=None, minlength=0):
    """Wrapper of `numpy.bincount`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    weights : DTensor
        Weights.
    minlength : int, optional
        Minimum number of bins, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.bitwise_and)
def bitwise_and(input_, other):
    """Wrapper of `numpy.bitwise_and`

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.bitwise_not)
def bitwise_not(input_):
    """Wrapper of `numpy.bitwise_not`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.bitwise_or)
def bitwise_or(input_, other):
    """Wrapper of `numpy.bitwise_or`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.bitwise_xor)
def bitwise_xor(input_, other):
    """Wrapper of `numpy.bitwise_xor`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_dispatch(np.can_cast)
def can_cast(from_, to):
    """Wrapper of `numpy.can_cast`.

    Parameters
    ----------
    from_ : data-type
        Data type to cast from.
    to : data-type
        Data type to cast to.
    """
    pass


@_wrap_ret
@_dispatch(np.ceil)
def ceil(input_):
    """Wrapper of `numpy.ceil`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.concatenate, sparse.concatenate)
def concatenate(inputs, axis=None):
    """Wrapper of `numpy.concatenate` and `sparse.concatenate`

    Parameters
    ----------
    inputs : sequence of STensor/DTensor
        Tensor to be concatenated.
    axis : int or None, optional
        Axis to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.conj, sparse.COO.conj)
def conj(input_):
    """Wrapper of `numpy.conj` and `sparse.COO.conj`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.cos)
def cos(input_):
    """Wrapper of `numpy.cos`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.cosh)
def cosh(input_):
    """Wrapper of `numpy.cosh`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.cumprod)
def cumprod(input_, axis=None, dtype=None):
    """Wrapper of `numpy.cumprod`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int, optional
        Axis to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.cumsum)
def cumsum(input_, axis=None, dtype=None):
    """Wrapper of `numpy.cumsum`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int, optional
        Axis to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.diag)
def diag(input_, k=0):
    """Wrapper of `numpy.diag`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    k : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.diagflat)
def diagflat(input_, k=0):
    """Wrapper of `numpy.diagflat`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.diagonal, sparse.diagonal)
def diagonal(input_, offset=0, axis1=0, axis2=1):
    """Wrapper of `numpy.diagonal`.

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
    pass


@_wrap_ret
@_dispatch(np.dot, sparse.dot)
def dot(input_, other):
    """Wrapper of `numpy.dot` and `sparse.dot`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The first operand.
    other : DTensor or STensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.empty)
def empty(shape, dtype):
    """Wrapper of `numpy.empty`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of tensor.
    dtype : data-type
        Data type of tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.empty_like)
def empty_like(input_, dtype):
    """Wrapper of `numpy.empty_like`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    dtype : data-type
        Data type of output.
    """
    pass


@_wrap_ret
@_dispatch(np.equal)
def equal(input_, other):
    """Wrapper of `numpy.equal`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.exp)
def exp(input_):
    """Wrapper of `numpy.exp`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.expm1, np.expm1)
def expm1(input_):
    """Wrapper of `numpy.expm1`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.eye, sparse.eye)
def eye(n, m=None, dtype=None):
    """Wrapper of `numpy.eye`.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of cols, by default None (equal to `n`)
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.flip)
def flip(input_, axis=None):
    """Wrapper of `numpy.flip`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.floor)
def floor(input_):
    """Wrapper of `numpy.floor`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.floor_divide)
def floor_divide(input_, other):
    """Wrapper of `numpy.floor_divide`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.fmod)
def fmod(input_, other):
    """Wrapper of `numpy.fmod`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.full, sparse.full)
def full(shape, value, dtype=None):
    """Wrapper of `numpy.full`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    value : scalar
        Fill value of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.full_like, sparse.full_like)
def full_like(input_, value, dtype=None):
    """Wrapper of `numpy.full_like` and `sparse.full_like`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The input tensor.
    value : scalar
        Fill value of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.imag)
def imag(input_):
    """Wrapper of `numpy.imag`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.isfinite)
def isfinite(input_):
    """Wrapper of `numpy.isinfinite`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.isinf)
def isinf(input_):
    """Wrapper of `numpy.isinf`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.isnan)
def isnan(input_):
    """Wrapper of `numpy.isnan`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.linspace)
def linspace(start, end, step, dtype=None):
    """Wrapper of `numpy.linspace`.

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
    pass


@_wrap_ret
@_dispatch(np.log)
def log(input_):
    """Wrapper of `numpy.log`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.log10)
def log10(input_):
    """Wrapper of `numpy.log10`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.log1p, np.log1p)
def log1p(input_):
    """Wrapper of `numpy.log1p`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.log2)
def log2(input_):
    """Wrapper of `numpy.log2`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.logical_and)
def logical_and(input_, other):
    """Wrapper of `numpy.logical_and`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.logical_not)
def logical_not(input_):
    """Wrapper of `numpy.logical_not`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.logical_or)
def logical_or(input_, other):
    """Wrapper of `numpy.logical_or`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.logical_xor)
def logical_xor(input_, other):
    """Wrapper of `numpy.logical_xor`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.logspace)
def logspace(start, stop, step, base=10, dtype=None):
    """Wrapper of `numpy.logspace`.

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
    pass


@_wrap_ret
@_dispatch(np.matmul, sparse.matmul)
def matmul(input_, other):
    """Wrapper of `numpy.matmul`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The first operand.
    other : DTensor or STensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.mean, sparse.COO.mean)
def mean(input_, axis=None, keepdims=False):
    """Wrapper of `numpy.mean` and `sparse.COO.mean`.

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
    pass


@_wrap_ret
@_dispatch(np.median)
def median(input_, axis=-1, keepdims=False):
    """Wrapper of `numpy.median`.

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
    pass


@_wrap_ret
@_dispatch(np.meshgrid)
def meshgrid(*inputs):
    """Wrapper of `numpy.meshigrid`.
    """
    pass


@_wrap_ret
@_dispatch(np.nonzero, sparse.COO.nonzero)
def nonzero(input_):
    """Wrapper of `numpy.nonzero` and `sparse.COO.nonzero`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.ones, sparse.ones)
def ones(shape, dtype=None):
    """Wrapper of `numpy.ones`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.ones_like, sparse.ones_like)
def ones_like(input_, dtype=None):
    """Wrapper of `numpy.ones_like` and `sparse.ones_like`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.prod, sparse.COO.prod)
def prod(input_, axis=None, keepdims=False, dtype=None):
    """Wrapper of `numpy.prod` and `sparse.COO.prod`.

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
    pass


@_wrap_ret
@_dispatch(np.real)
def real(input_):
    """Wrapper of `numpy.real`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.reciprocal)
def reciprocal(input_):
    """Wrapper of `numpy.reciprocal`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.remainder)
def remainder(input_, other):
    """Wrapper of `numpy.remainder`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.reshape, sparse.COO.reshape)
def reshape(input_, shape):
    """Wrapper of `numpy.rehsape` and `sparse.COO.reshape`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    shape : tuple of ints
        Shape of new tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.roll)
def roll(input_, shift, axis=None):
    """Wrapper of `numpy.roll`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    shift : int or tuple of ints
        Shift numbers of each axis or axes.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.rot90)
def rot90(input_, k=1, axes=(0, 1)):
    """Wrapper of `numpy.rot90`

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    k : int, optional
        Number of rotation times, by default 1
    axes : tuple, optional
        The axes in which the input is rotated, by default (0, 1)
    """
    pass


@_wrap_ret
@_dispatch(np.sign)
def sign(input_):
    """Wrapper of `numpy.sign`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.sin, np.sin)
def sin(input_):
    """Wrapper of `numpy.sin`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.sinh, np.sinh)
def sinh(input_):
    """Wrapper of `numpy.sinh`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.split)
def split(input_, indices_or_sections, axis=0):
    """Wrapper of `numpy.split`.

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
    pass


@_wrap_ret
@_dispatch(np.sqrt, np.sqrt)
def sqrt(input_):
    """Wrapper of `numpy.sqrt`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.square, np.square)
def square(input_):
    """Wrapper of `numpy.square`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.squeeze)
def squeeze(input_, axis=None):
    """Wrapper of `numpy.squeeze`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.stack, sparse.stack)
def stack(inputs, axis=0):
    """Wrapper of `numpy.stack` and `sparse.stack`.

    Parameters
    ----------
    inputs : DTensor and STensor.
        Input tensor.
    axis : int, optional
        Axis to operate on, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.std, sparse.COO.std)
def std(input_, axis=None, keepdims=False):
    """Wrapper of `numpy.std` and `sparse.COO.std`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    pass


@_wrap_ret
@_dispatch(np.sum, sparse.COO.sum)
def sum(input_, axis=None, dtype=None, keepdims=False):
    """Wrapper of `numpy.sum` and `sparse.COO.sum`.

    Parameters
    ----------
    input_ : DTensor or STensor.
        Input tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes to operate on, by default None
    dtype : data-type, optional
        Data type of output, by default None
    keepdims : bool, optional
        If true, the axes along which the operation performed are left to size
        one, by default False
    """
    pass


@_wrap_ret
@_dispatch(np.take)
def take(input_, indices):
    """Wrapper of `numpy.take`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    indices : DTensor
        Indices of extracted values.
    """
    pass


@_wrap_ret
@_dispatch(np.tan, np.tan)
def tan(input_):
    """Wrapper of `numpy.tan`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.tanh, np.tanh)
def tanh(input_):
    """Wrapper of `numpy.tanh`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.tensordot, sparse.tensordot)
def tensordot(input_, other, axes=2):
    """Wrapper of `numpy.tensordot` and `sparse.tensordot`.

    Parameters
    ----------
    input_ : DTensor or STensor
        The first operand.
    other : DTensor or STensor
        The second operand.
    axes : int or tuple of ints, optional
        Axis or axes along which the operation is performed, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.trace, np.trace)
def trace(input_):
    """Wrapper of `numpy.trace`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.transpose, sparse.COO.transpose)
def transpose(input_, axes=None):
    """Wrapper of `numpy.transpose` and `sparse.COO.transpose`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor
    axes : list of ints, optional
       Axes along which the operation is performed, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.tril, sparse.tril)
def tril(input_, k=0):
    """Wrapper of `numpy.tril` and `sparse.tril`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    k : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.tril_indices)
def tril_indices(n, m=0, k=0):
    """Wrapper of `numpy.tril_indices`.

    Parameters
    ----------
    n : int
        Number of row of output tensor.
    m : int, optional
        Number of column of output tensor, by default 0
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.triu, sparse.triu)
def triu(input_, k=0):
    """Wrapper of `numpy.triu` and `sparse.triu`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor
    k : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.triu_indices)
def triu_indices(n, m=0, k=0):
    """Wrapper of `numpy.triu_indices`.

    Parameters
    ----------
    n : int
        Number of row of output tensor.
    m : int, optional
        Number of column of output tensor, by default 0
    offset : int, optional
        Offset to main diagonal, by default 0
    """
    pass


@_wrap_ret
@_dispatch(np.true_divide)
def true_divide(input_, other):
    """Wrapper of `numpy.true_divide`.

    Parameters
    ----------
    input_ : DTensor
        The first operand.
    other : DTensor
        The second operand.
    """
    pass


@_wrap_ret
@_dispatch(np.trunc)
def trunc(input_):
    """Wrapper of `numpy.trunc`.

    Parameters
    ----------
    input_ : DTensor
        Input dense tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.unique)
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    """Wrapper of `numpy.unique`.

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
    pass


@_wrap_ret
@_dispatch(np.var, sparse.COO.var)
def var(input_, axis=None, keepdims=False):
    """Wrapper of `numpy.var` and `sparse.COO.var`.

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
    pass


@_wrap_ret
@_dispatch(np.where, sparse.where)
def where(condition, x, y):
    """Wrapper of `numpy.where` and `sparse.where`.

    Parameters
    ----------
    condition : DTensor of bool
        Where True, yield x, otherwise yield y.
    x : DTensor or STensor
        The first tensor.
    y : DTensor or STensor
        The second tensor.
    """
    pass


@_wrap_ret
@_dispatch(np.zeros, sparse.zeros)
def zeros(shape, dtype=None):
    """Wrapper of `numpy.zeros` and `sparse.zeros`.

    Parameters
    ----------
    shape : tuple of ints
        Shape of output tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass


@_wrap_ret
@_dispatch(np.zeros_like, sparse.zeros_like)
def zeros_like(input_, dtype=None):
    """Wrapper of `numpy.zeros_like` and `sparse.zeros_like`.

    Parameters
    ----------
    input_ : DTensor or STensor
        Input tensor.
    dtype : data-type, optional
        Data type of output tensor, by default None
    """
    pass
