from typing import Union
import numpy as np

from .tensor import DTensor, STensor

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


def _wrap_ret(ret):
    if isinstance(ret, np.ndarray):
        return DTensor(ret)
    else:
        return ret


def _check_dense(input_, func, other=None):
    if not isinstance(input_, DTensor):
        raise TypeError(
            f"`st.{func.__name__}` does not support {type(input_)} type.")
    if other is not None:
        if not isinstance(other, DTensor):
            raise TypeError(
                f"`st.{func.__name__}` does not support {type(other)} type.")


def _check_param(input_, func):
    if isinstance(input_, STensor):
        return STensor
    elif isinstance(input_, DTensor):
        return DTensor
    else:
        raise TypeError(
            f"`st.{func.__name__}` does not support {type(input_)} type.")


def _check_params(input_, other, func):
    if isinstance(input_, STensor) and isinstance(other, STensor):
        return STensor
    elif isinstance(input_, DTensor) and isinstance(other, DTensor):
        return DTensor
    else:
        raise TypeError(
            f"`st.{func.__name__}` does not support {type(input_)} and {type(other)} type.")


def add(input_, other):
    type_ = _check_params(input_, other, add)
    return type_(np.add(input_.data, other.data))


def all(input_, axis=None, keepdim=False):
    _check_dense(input_, all)
    ret = np.all(input_.data, axis=axis, keepdims=keepdim)
    return _wrap_ret(ret)


def allclose(input_, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    _check_dense(input_, allclose, other)
    return np.allclose(input_.data, other.data, rtol, atol, equal_nan)


def angle(input_, deg=False):
    _check_dense(input_, angle)
    ret = np.angle(input_.data, deg)
    return _wrap_ret(ret)


def any(input_, axis, keepdim=False):
    _check_dense(input_, any)
    ret = np.any(input_.data, axis, keepdim)
    return _wrap_ret(ret)


def arange(start, stop, step, dtype=None):
    return DTensor(np.arange(start, stop, step, dtype=dtype))


def argmax(input_, axis=None) -> DTensor:
    type_ = _check_param(input_, argmax)
    ret = np.argmax(input_.data, axis)
    if type_ == STensor:
        ret = np.array(ret).flatten()
    return DTensor(ret)


def argmin(input_, axis=None) -> DTensor:
    type_ = _check_param(input_, argmin)
    ret = np.argmin(input_.data, axis)
    if type_ == STensor:
        ret = np.array(ret).flatten()
    return DTensor(ret)


def argsort(input_: DTensor, axis=-1) -> DTensor:
    _check_dense(input_, argsort)
    return DTensor(np.argsort(input_.data, axis))


def bincount(input_: DTensor, weights, minlength=0) -> DTensor:
    _check_dense(input_, bincount)
    return DTensor(np.bincount(input_.data, weights, minlength))


def bitwise_and(input_: DTensor, other: DTensor) -> DTensor:
    _check_dense(input_, bitwise_and, other)
    ret = np.bitwise_and(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_not(input_, other):
    _check_dense(input_, bitwise_and, other)
    ret = np.bitwise_not(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_or(input_, other):
    _check_dense(input_, bitwise_or, other)
    ret = np.bitwise_or(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_xor(input_, other):
    _check_dense(input_, bitwise_xor, other)
    ret = np.bitwise_xor(input_.data, other.data)
    return _wrap_ret(ret)


def can_cast(from_, to):
    return np.can_cast(from_, to)


def ceil(input_):
    _check_dense(input_, ceil)
    return DTensor(np.ceil(input_.data))


def conj(input_):
    type_ = _check_param(input_, conj)
    return type_(np.conj(input_.data))


def cos(input_):
    _check_dense(input_, cos)
    return DTensor(np.cos(input_.data))


def cosh(input_):
    _check_dense(input_, cosh)
    return DTensor(np.cos(input_.data))


def cross(input_, other, axis=-1):
    _check_dense(input_, cross, other)
    return DTensor(np.cross(input_.data, other.data, axis=axis))


def cumprod(input_, axis=None, dtype=None):
    _check_dense(input_, cumprod)
    return DTensor(np.cumprod(input_.data, axis=axis, dtype=dtype))


def cumsum(input_, axis=None, dtype=None):
    _check_dense(input_, cumsum)
    return DTensor(np.cumsum(input_.data, axis, dtype))


def diag(input_, k):
    _check_dense(input_, diag)
    return DTensor(np.diag(input_.data, k))


def diagflat(input_, offset=0):
    _check_dense(input_, diagflat)
    return DTensor(np.diagflat(input_.data, offset))


def diagonal(input_, offset, axis1=None, axis2=None):
    _check_dense(input_, diagonal)
    return DTensor(np.diagonal(input_.data, offset, axis1, axis2))


def dot(input_, other):
    type_ = _check_params(input_, other, bitwise_and)
    return type_(np.dot(input_.data, other.data))


def einsum(equation, *operands):
    return DTensor(np.eigsum(equation, *operands))


def empty(shape, dtype):
    return DTensor(np.empty(shape, dtype=dtype))


def empty_like(input_, dtype):
    _check_dense(input_, empty_like)
    return DTensor(np.empty_like(input_.data, dtype=dtype))


def equal(input_, other):
    _check_dense(input_, equal, other)
    return DTensor(np.equal(input_.data, other.data))


def exp(input_):
    _check_dense(input_, exp)
    return DTensor(np.exp(input_.data))


def expm1(input_):
    type_ = _check_param(input_, expm1)
    return type_(np.expm1(input_.data))


def eye(n, m=None, dtype=None):
    return DTensor(np.eye(n, m, dtype=dtype))


def flip(input_, axis=None):
    type_ = _check_param(input_, flip)
    return type_(np.flip(input_.data, axis))


def floor(input_):
    _check_dense(input_, floor)
    return DTensor(np.floor(input_.data))


def floor_divide(input_, other):
    _check_dense(input_, floor_divide, other)
    return DTensor(np.floor_divide(input_.data, other.data))


def fmod(input_, other):
    _check_dense(input_, fmod, other)
    return DTensor(np.fmod(input_.data, other.data))


def full(shape, value, dtype=None):
    return DTensor(np.full(shape, value, dtype=dtype))


def full_like(input_, value, dtype=None):
    _check_dense(input_, full_like)
    return DTensor(np.full_like(input_.data, value, dtype=dtype))


def imag(input_):
    type_ = _check_param(input_, imag)
    return type_(np.imag(input_.data))


def isfinite(input_):
    _check_dense(input_, isfinite)
    return DTensor(np.isfinite(input_.data))


def isinf(input_):
    _check_dense(input_, isinf)
    return DTensor(np.isinf(input_.data))


def isnan(input_):
    _check_dense(input_, isnan)
    return DTensor(np.isnan(input_.data))


def linspace(start, end, step, dtype=None):
    return DTensor(np.linspace(start, end, step, dtype=dtype))


def log(input_):
    _check_dense(input_, log)
    return DTensor(np.log(input_.data))


def log10(input_):
    _check_dense(input_, log10)
    return DTensor(np.log10(input_.data))


def log1p(input_):
    _check_dense(input_, log1p)
    return DTensor(np.log1p(input_.data))


def log2(input_):
    _check_dense(input_, log2)
    return DTensor(np.log2(input_.data))


def logical_and(input_, other):
    _check_dense(input_, logical_and, other)
    return DTensor(np.logical_and(input_.data, other.data))


def logical_not(input_):
    _check_dense(input_, logical_not)
    return DTensor(np.logical_not(input_.data))


def logical_or(input_, other):
    _check_dense(input_, logical_or, other)
    return DTensor(np.logical_or(input_.data, other.data))


def logical_xor(input_, other):
    _check_dense(input_, logical_xor, other)
    return DTensor(np.logical_xor(input_.data, other.data))


def logspace(start, stop, step, base=10, dtype=None):
    return DTensor(np.logspace(start, stop, step, base=base, dtype=dtype))


def matmul(input_, other):
    _check_dense(input_, matmul, other)
    return DTensor(np.matmul(input_.data, other.data))


def mean(input_, axis=None, keepdim=False):
    type_ = _check_param(input_, mean)
    ret = np.mean(input_.data, axis, keepdims=keepdim)
    if type_ == STensor:
        ret = np.array(ret).flatten()
    return DTensor(ret)


def median(input_, dim=-1, keepdim=False):
    type_ = _check_param(input_, median)
    ret = np.median(input_.data, axis, keepdims=keepdim)
    if type_ == STensor:
        ret = np.array(ret).flatten()
    return DTensor(ret)


def meshgrid(*inputs):
    datas = [i.data for i in inputs]
    return tuple([DTensor(d) for d in np.meshgrid(*datas)])


def nonzero(input_):
    return tuple([DTensor(d) for d in np.nonzero(input_.data)])


def ones(shape, dtype=None):
    return DTensor(np.ones(shape, dtype=dtype))


def ones_like(input_, dtype=None):
    _check_dense(input_, ones_like)
    return DTensor(np.ones_like(input_.data, dtype=dtype))


def prod(input_, axis=None, keepdim=False, dtype=None):
    _check_dense(input_, prod)
    ret = np.prod(input_.data, axis=axis, keepdims=keepdim, dtype=dtype)
    return _wrap_ret(ret)


def real(input_):
    type_ = _check_param(input_, real)
    return type_(np.real(input_.data))


def reciprocal(input_):
    _check_dense(input_, reciprocal)
    return DTensor(np.reciprocal(input_.data))


def remainder(input_, other):
    _check_dense(input_, remainder, other)
    return DTensor(np.remainder(input_.data, other.data))


def reshape(input_, shape):
    type_ = _check_param(input_, reshape)
    return type_(np.rehsape(input_.data, shape))


def roll(input_, shift, axis=None):
    _check_dense(input_, roll)
    return DTensor(np.roll(input_.data, shift, axis=axis))


def rot90(input_, k=1, axes=(0, 1)):
    _check_dense(innput_, rot90)
    return DTensor(np.rot90(input_.data, k, axes))


def round(input_):
    type_ = _check_param(input_, round)
    return type_(np.round(input_.data))


def sign(input_):
    _check_dense(input_, sign)
    return DTensor(np.sign(input_.data))


def sin(input_):
    type_ = _check_param(input_, sin)
    return type_(np.sin(input_.data))


def sinh(input_):
    type_ = _check_param(input_, sinh)
    return type_(np.sinh(input_.data))


def sort(input_, axis=-1):
    _check_dense(input_, sort)
    return DTensor(np.sort(input_.data, axis=axis))


def split(input_, indices_or_sections, axis=0):
    _check_dense(input_, split)
    return DTensor(np.split(input_.data, indices_or_sections, axis))


def sqrt(input_):
    type_ = _check_param(input_, sqrt)
    return type_(np.sqrt(input_.data))


def square(input_):
    type_ = _check_param(input_, square)
    return type_(np.square(input_.data))


def squeeze(input_, axis=None):
    _check_dense(input_, squeeze)
    return DTensor(np.squeeze(input_.data, axis=axis))


def stack(inputs, axis=0):
    return DTensor(np.stack(inputs, axis))


def std(input_, axis=None, keepdim=False):
    _check_dense(input_, std)
    return DTensor(np.std(input_.data, axis=axis, dtype=dtype, keepdims=keepdim))


def sum(input_, axis=None, dtype=None, keepdim=False):
    ret = np.sum(input_.data, axis=axis, dtype=dtype, keepdims=keepdim)
    return _wrap_ret(ret)


def take(input_, indices):
    _check_dense(input_, take)
    return DTensor(np.take(input_.data, indices))


def tan(input_):
    type_ = _check_param(input_, tan)
    return type_(np.tan(input_.data))


def tanh(input_):
    type_ = _check_param(input_, tan)
    return type_(np.tanh(input_.data))


def tensordot(input_, other, axes=2):
    _check_dense(input_, tensordot, other)
    return DTensor(np.tensordot(input_.data, other.data, axes))


def trace(input_):
    _check_dense(input_)
    ret = np.trace(input_.data)
    return _wrap_ret(ret)


def transpose(input_, axes=None):
    type_ = _check_param(input_, transpose)
    return type_(np.transpose(input_.data, axes))


def tril(input_, k=0):
    _check_dense(input_, tril)
    return DTensor(np.tril(input_.data, k))


def tril_indices(n, m=0, offset=0):
    ret = np.tril_indices(n, k=offset, m=m)
    return tuple([DTensor(index) for index in ret])


def triu(input_, k=0):
    _check_dense(input_, triu)
    return DTensor(np.triu(input_.data, k))


def triu_indices(n, m=0, offset=0):
    ret = np.triu_indices(n, k=offset, m=m)
    return tuple([DTensor(index) for index in ret])


def true_divide(input_, other):
    return DTensor(np.true_divide(input_.data, other.data))


def trunc(input_):
    _check_dense(input_, trunc)
    return DTensor(np.trunc(input_.data))


def unique(input_, return_inverse=False, return_counts=False, axis=None):
    _check_dense(input_, unique)
    return DTensor(np.unique(input_.data, return_inverse=return_inverse, return_counts=return_counts, axis=axis))


def var(input_, axis=None, keepdim=False):
    _check_dense(input_, var)
    ret = np.var(input_.data, axis, keepdims=keepdim)
    return _wrap_ret(ret)


def where(condition, x, y):
    _check_dense(x, where, y)
    return DTensor(np.where(condition, x, y))


def zeros(shape, dtype=None):
    return DTensor(np.zeros(shape, dtype=dtype))


def zeros_like(input_, dtype=None):
    _check_dense(input_, zeros_like)
    return DTensor(np.zeros_like(input_.data, dtype=dtype))
