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
    return torch.add(input_._data, other._data)


@_wrap_ret
def all(input_, axis=None, keepdims=False):
    if axis is None:
        return torch.all(input_._data)
    return torch.all(input_._data, dim=axis, keepdim=keepdims)


@_wrap_ret
def allclose(input_, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(input_._data, other._data, rtol, atol, equal_nan)


@_wrap_ret
def angle(input_, deg=False):
    if deg:
        ret = torch.angle(input_) * 180 / math.pi
    else:
        ret = torch.angle(input_)
    return ret


@_wrap_ret
def any(input_, axis=None, keepdims=False):
    if axis is None:
        return torch.any(input_._data)
    return torch.any(input_._data, dim=axis, keepdim=keepdims)


@_wrap_ret
def arange(start, stop, step, dtype=None):
    return torch.arange(start, stop, step, dtype=dtype)


@_wrap_ret
def argmax(input_, axis=None):
    if axis is None:
        return torch.argmin(input_)
    else:
        return torch.argmax(input_, dim=axis)


@_wrap_ret
def argmin(input_, axis=None):
    if axis is None:
        return torch.argmin(input_)
    else:
        return torch.argmax(input_, dim=axis)


@_wrap_ret
def argsort(input_, axis=-1):
    return torch.argsort(input_._data, dim=axis)


@_wrap_ret
def bincount(input_, weights, minlength=0):
    return torch.bincount(input_._data, weights, minlength)


@_wrap_ret
def bitwise_and(input_, other):
    return torch.bitwise_and(input_._data, other._data)


@_wrap_ret
def bitwise_not(input_, other):
    return torch.bitwise_not(input_._data, other._data)


@_wrap_ret
def bitwise_or(input_, other):
    return torch.bitwise_or(input_._data, other._data)


@_wrap_ret
def bitwise_xor(input_, other):
    return torch.bitwise_xor(input_._data, other._data)


def can_cast(from_, to):
    return torch.can_cast(from_, to)


@_wrap_ret
def ceil(input_):
    return torch.ceil(input_._data)


@_wrap_ret
def conj(input_):
    return torch.conj(input_._data)


@_wrap_ret
def cos(input_):
    return torch.cos(input_._data)


@_wrap_ret
def cosh(input_):
    return torch.cos(input_._data)


@_wrap_ret
def cross(input_, other, axis=-1):
    return torch.cross(input_._data, other._data, dim=axis)


@_wrap_ret
def cumprod(input_, axis=None, dtype=None):
    return torch.cumprod(input_._data, dim=axis, dtype=dtype)


@_wrap_ret
def cumsum(input_, axis=None, dtype=None):
    return torch.cumsum(input_._data, axis, dtype)


@_wrap_ret
def diag(input_, k):
    return torch.diag(input_._data, k)


@_wrap_ret
def diagflat(input_, offset=0):
    return torch.diagflat(input_._data, offset)


@_wrap_ret
def diagonal(input_, offset=0, axis1=0, axis2=1):
    return torch.diagonal(input_._data, offset=offset, dim1=axis1, dim2=axis2)


@_wrap_ret
def dot(input_, other):
    if input_._data.ndim == 1 and other._data.ndim == 1:
        return torch.dot(input_._data, other._data)
    return torch.matmul(input_._data, other._data)


@_wrap_ret
def einsum(equation, *operands):
    return torch.eigsum(equation, *operands)


@_wrap_ret
def empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


@_wrap_ret
def empty_like(input_, dtype):
    return torch.empty_like(input_._data, dtype=dtype)


@_wrap_ret
def equal(input_, other):
    return torch.equal(input_._data, other._data)


@_wrap_ret
def exp(input_):
    return torch.exp(input_._data)


@_wrap_ret
def expm1(input_):
    return torch.expm1(input_._data)


@_wrap_ret
def eye(n, m=None, dtype=None):
    return torch.eye(n, m, dtype=dtype)


@_wrap_ret
def flip(input_, axis=None):
    return torch.flip(input_._data, axis)


@_wrap_ret
def floor(input_):
    return torch.floor(input_._data)


@_wrap_ret
def floor_divide(input_, other):
    return torch.floor_divide(input_._data, other._data)


@_wrap_ret
def fmod(input_, other):
    return torch.fmod(input_._data, other._data)


@_wrap_ret
def full(shape, value, dtype=None):
    return torch.full(shape, value, dtype=dtype)


@_wrap_ret
def full_like(input_, value, dtype=None):
    return torch.full_like(input_._data, value, dtype=dtype)


@_wrap_ret
def imag(input_):
    return torch.imag(input_._data)


@_wrap_ret
def isfinite(input_):
    return torch.isfinite(input_._data)


@_wrap_ret
def isinf(input_):
    return torch.isinf(input_._data)


@_wrap_ret
def isnan(input_):
    return torch.isnan(input_._data)


@_wrap_ret
def linspace(start, end, step, dtype=None):
    return torch.linspace(start, end, step, dtype=dtype)


@_wrap_ret
def log(input_):
    return torch.log(input_._data)


@_wrap_ret
def log10(input_):
    return torch.log10(input_._data)


@_wrap_ret
def log1p(input_):
    return torch.log1p(input_._data)


@_wrap_ret
def log2(input_):
    return torch.log2(input_._data)


@_wrap_ret
def logical_and(input_, other):
    return torch.logical_and(input_._data, other._data)


@_wrap_ret
def logical_not(input_):
    return torch.logical_not(input_._data)


@_wrap_ret
def logical_or(input_, other):
    return torch.logical_or(input_._data, other._data)


@_wrap_ret
def logical_xor(input_, other):
    return torch.logical_xor(input_._data, other._data)


@_wrap_ret
def logspace(start, stop, step, base=10, dtype=None):
    return torch.logspace(start, stop, step, base=base, dtype=dtype)


@_wrap_ret
def matmul(input_, other):
    return torch.matmul(input_._data, other._data)


@_wrap_ret
def mean(input_, axis=None, keepdims=False):
    if axis is None:
        ret = torch.mean(input_._data)
    else:
        ret = torch.mean(input_._data, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def median(input_, axis=-1, keepdims=False):
    if axis is None:
        ret = torch.median(input_._data)
    else:
        ret = torch.median(input_._data, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def meshgrid(*inputs):
    datas = [i._data for i in inputs]
    return tuple([d for d in torch.meshgrid(*datas)])


@_wrap_ret
def nonzero(input_):
    return tuple([d for d in torch.nonzero(input_._data, as_tuple=True)])


@_wrap_ret
def ones(shape, dtype=None):
    return torch.ones(shape, dtype=dtype)


@_wrap_ret
def ones_like(input_, dtype=None):
    return torch.ones_like(input_._data, dtype=dtype)


@_wrap_ret
def prod(input_, axis=None, keepdims=False, dtype=None):
    if axis is None:
        return torch.prod(input_._data, dtype=dtype)
    return torch.prod(input_._data, dim=axis, keepdim=keepdims, dtype=dtype)


@_wrap_ret
def real(input_):
    return torch.real(input_._data)


@_wrap_ret
def reciprocal(input_):
    return torch.reciprocal(input_._data)


@_wrap_ret
def remainder(input_, other):
    return torch.remainder(input_._data, other._data)


@_wrap_ret
def reshape(input_, shape):
    return torch.rehsape(input_._data, shape)


@_wrap_ret
def roll(input_, shift, axis=None):
    return torch.roll(input_._data, shift, dims=axis)


@_wrap_ret
def rot90(input_, k=1, axes=(0, 1)):
    return torch.rot90(input_._data, k, dims=axes)


@_wrap_ret
def round(input_):
    return torch.round(input_._data)


@_wrap_ret
def sign(input_):
    return torch.sign(input_._data)


@_wrap_ret
def sin(input_):
    return torch.sin(input_._data)


@_wrap_ret
def sinh(input_):
    return torch.sinh(input_._data)


@_wrap_ret
def split(input_, indices_or_sections, axis=0):
    return torch.split(input_._data, indices_or_sections, axis)


@_wrap_ret
def sqrt(input_):
    return torch.sqrt(input_._data)


@_wrap_ret
def square(input_):
    return torch.square(input_._data)


@_wrap_ret
def squeeze(input_, axis=None):
    return torch.squeeze(input_._data, dim=axis)


@_wrap_ret
def stack(inputs, axis=0):
    return torch.stack(inputs, axis)


@_wrap_ret
def std(input_, axis=None, keepdims=False):
    if axis is None:
        ret = torch.std(input_._data, unbiased=False)
    else:
        ret = torch.std(input_._data, dim=axis,
                        keepdim=keepdims, unbiased=False)
    return ret


@_wrap_ret
def sum(input_, axis=None, dtype=None, keepdims=False):
    if axis is None:
        ret = torch.sum(input_._data, dtype=dtype)
    else:
        ret = torch.sum(input_._data, dim=axis, dtype=dtype, keepdim=keepdims)
    return ret


@_wrap_ret
def take(input_, indices):
    return torch.take(input_._data, indices)


@_wrap_ret
def tan(input_):
    return torch.tan(input_._data)


@_wrap_ret
def tanh(input_):
    return torch.tanh(input_._data)


@_wrap_ret
def tensordot(input_, other, axes=2):
    return torch.tensordot(input_._data, other._data, axes)


@_wrap_ret
def trace(input_):
    return torch.trace(input_._data)


@_wrap_ret
def transpose(input_, axes=None):
    if axes is None:
        axes = (0, 1)
    return torch.transpose(input_, axes[0], axes[1])


@_wrap_ret
def tril(input_, k=0):
    return torch.tril(input_._data, k)


@_wrap_ret
def tril_indices(n, m=0, offset=0):
    return torch.tril_indices(row=m, col=m, offset=offset)


def triu(input_, k=0):
    return torch.triu(input_._data, k)


@_wrap_ret
def triu_indices(n, m=0, offset=0):
    ret = torch.triu_indices(row=m, col=m, offset=offset)
    return tuple([index for index in ret])


@_wrap_ret
def true_divide(input_, other):
    return torch.true_divide(input_._data, other._data)


@_wrap_ret
def trunc(input_):
    return torch.trunc(input_._data)


@_wrap_ret
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    return torch.unique(input_._data, return_inverse=return_inverse,
                        return_counts=return_counts, dim=axis)


@_wrap_ret
def var(input_, axis=None, keepdims=False):
    if axis is None:
        ret = torch.var(input_)
    else:
        ret = torch.var(input_, dim=axis, keepdim=keepdims)
    return ret


@_wrap_ret
def where(condition, x, y):
    return torch.where(condition, x, y)


@_wrap_ret
def zeros(shape, dtype=None):
    return torch.zeros(shape, dtype=dtype)


@_wrap_ret
def zeros_like(input_, dtype=None):
    return torch.zeros_like(input_._data, dtype=dtype)
