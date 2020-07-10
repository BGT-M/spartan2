from typing import Union
import functools
import numpy as np
import scipy.sparse as ssp

from .tensor import DTensor, STensor, _wrap_ret, _require_dense, _check_params

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


@_wrap_ret()
@_check_params(0, 1)
def add(input_, other):
    return np.add(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def all(input_, axis=None, keepdim=False):
    return np.all(input_._data, axis=axis, keepdims=keepdim)


@_wrap_ret()
@_require_dense(0, 1)
def allclose(input_, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    return np.allclose(input_._data, other._data, rtol, atol, equal_nan)


@_wrap_ret()
@_require_dense(0)
def angle(input_, deg=False):
    return np.angle(input_._data, deg)


@_wrap_ret()
@_require_dense(0)
def any(input_, axis, keepdim=False):
    return np.any(input_._data, axis, keepdim)


@_wrap_ret()
def arange(start, stop, step, dtype=None):
    return np.arange(start, stop, step, dtype=dtype)


@_wrap_ret(squeeze=True)
@_check_params(0)
def argmax(input_, axis=None) -> DTensor:
    return np.argmin(input_._data, axis)


@_wrap_ret(squeeze=True)
@_check_params(0)
def argmin(input_, axis=None) -> DTensor:
    return np.argmin(input_._data, axis)


@_wrap_ret()
@_require_dense(0)
def argsort(input_: DTensor, axis=-1) -> DTensor:
    return DTensor(np.argsort(input_._data, axis))


@_wrap_ret()
@_require_dense(0)
def bincount(input_: DTensor, weights, minlength=0) -> DTensor:
    return np.bincount(input_._data, weights, minlength)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_and(input_: DTensor, other: DTensor) -> DTensor:
    return np.bitwise_and(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_not(input_, other):
    return np.bitwise_not(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_or(input_, other):
    return np.bitwise_or(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_xor(input_, other):
    return np.bitwise_xor(input_._data, other._data)


def can_cast(from_, to):
    return np.can_cast(from_, to)


@_wrap_ret()
@_require_dense(0)
def ceil(input_):
    return np.ceil(input_._data)


@_wrap_ret()
def conj(input_):
    return np.conj(input_._data)


@_wrap_ret()
@_require_dense(0)
def cos(input_):
    return np.cos(input_._data)


@_wrap_ret()
@_require_dense(0)
def cosh(input_):
    return np.cos(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def cross(input_, other, axis=-1):
    return np.cross(input_._data, other._data, axis=axis)


@_wrap_ret()
@_require_dense(0)
def cumprod(input_, axis=None, dtype=None):
    return np.cumprod(input_._data, axis=axis, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def cumsum(input_, axis=None, dtype=None):
    return np.cumsum(input_._data, axis, dtype)


@_wrap_ret()
@_require_dense(0)
def diag(input_, k):
    return np.diag(input_._data, k)


@_wrap_ret()
@_require_dense(0)
def diagflat(input_, offset=0):
    return np.diagflat(input_._data, offset)


@_wrap_ret()
@_require_dense(0)
def diagonal(input_, offset, axis1=None, axis2=None):
    return np.diagonal(input_._data, offset, axis1, axis2)


@_wrap_ret()
@_check_params(0, 1)
def dot(input_, other):
    return np.dot(input_._data, other._data)


@_wrap_ret()
def einsum(equation, *operands):
    return np.eigsum(equation, *operands)


@_wrap_ret()
def empty(shape, dtype):
    return np.empty(shape, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def empty_like(input_, dtype):
    return np.empty_like(input_._data, dtype=dtype)


@_wrap_ret()
@_require_dense(0, 1)
def equal(input_, other):
    return np.equal(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def exp(input_):
    return np.exp(input_._data)


@_wrap_ret()
def expm1(input_):
    return np.expm1(input_._data)


@_wrap_ret()
def eye(n, m=None, dtype=None):
    return Dnp.eye(n, m, dtype=dtype)


@_wrap_ret()
@_check_params(0)
def flip(input_, axis=None):
    return np.flip(input_._data, axis)


@_wrap_ret()
@_require_dense(0)
def floor(input_):
    return np.floor(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def floor_divide(input_, other):
    return np.floor_divide(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def fmod(input_, other):
    return np.fmod(input_._data, other._data)


@_wrap_ret()
def full(shape, value, dtype=None):
    return np.full(shape, value, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def full_like(input_, value, dtype=None):
    return np.full_like(input_._data, value, dtype=dtype)


@_wrap_ret()
def imag(input_):
    return np.imag(input_._data)


@_wrap_ret()
@_require_dense(0)
def isfinite(input_):
    return np.isfinite(input_._data)


@_wrap_ret()
@_require_dense(0)
def isinf(input_):
    return np.isinf(input_._data)


@_wrap_ret()
@_require_dense(0)
def isnan(input_):
    return np.isnan(input_._data)


@_wrap_ret()
def linspace(start, end, step, dtype=None):
    return np.linspace(start, end, step, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def log(input_):
    return np.log(input_._data)


@_wrap_ret()
@_require_dense(0)
def log10(input_):
    return np.log10(input_._data)


@_wrap_ret()
def log1p(input_):
    return np.log1p(input_._data)


@_wrap_ret()
@_require_dense(0)
def log2(input_):
    return np.log2(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_and(input_, other):
    return np.logical_and(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def logical_not(input_):
    return np.logical_not(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_or(input_, other):
    return np.logical_or(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_xor(input_, other):
    return np.logical_xor(input_._data, other._data)


@_wrap_ret()
def logspace(start, stop, step, base=10, dtype=None):
    return np.logspace(start, stop, step, base=base, dtype=dtype)


@_wrap_ret()
@_require_dense(0, 1)
def matmul(input_, other):
    return np.matmul(input_._data, other._data)


@_wrap_ret()
def mean(input_, axis=None, keepdim=False):
    return np.mean(input_._data, axis, keepdims=keepdim)


@_wrap_ret()
def median(input_, axis=-1, keepdim=False):
    return np.median(input_._data, axis, keepdims=keepdim)


@_wrap_ret()
def meshgrid(*inputs):
    datas = [i._data for i in inputs]
    return tuple([DTensor(d) for d in np.meshgrid(*datas)])


@_wrap_ret()
def nonzero(input_):
    return tuple([DTensor(d) for d in np.nonzero(input_._data)])


@_wrap_ret()
def ones(shape, dtype=None):
    return np.ones(shape, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def ones_like(input_, dtype=None):
    return np.ones_like(input_._data, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def prod(input_, axis=None, keepdim=False, dtype=None):
    return np.prod(input_._data, axis=axis, keepdims=keepdim, dtype=dtype)


@_wrap_ret()
def real(input_):
    return np.real(input_._data)


@_wrap_ret()
@_require_dense(0)
def reciprocal(input_):
    return np.reciprocal(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def remainder(input_, other):
    return np.remainder(input_._data, other._data)


@_wrap_ret()
def reshape(input_, shape):
    return np.rehsape(input_._data, shape)


@_wrap_ret()
@_require_dense(0)
def roll(input_, shift, axis=None):
    return np.roll(input_._data, shift, axis=axis)


@_wrap_ret()
@_require_dense(0)
def rot90(input_, k=1, axes=(0, 1)):
    return np.rot90(input_._data, k, axes)


@_wrap_ret()
def round(input_):
    return np.round(input_._data)


@_wrap_ret()
@_require_dense(0)
def sign(input_):
    return np.sign(input_._data)


@_wrap_ret()
def sin(input_):
    return np.sin(input_._data)


@_wrap_ret()
def sinh(input_):
    return np.sinh(input_._data)


@_wrap_ret()
@_require_dense(0)
def sort(input_, axis=-1):
    return np.sort(input_._data, axis=axis)


@_wrap_ret()
@_require_dense(0)
def split(input_, indices_or_sections, axis=0):
    return np.split(input_._data, indices_or_sections, axis)


@_wrap_ret()
def sqrt(input_):
    return np.sqrt(input_._data)


@_wrap_ret()
def square(input_):
    return np.square(input_._data)


@_wrap_ret()
@_require_dense(0)
def squeeze(input_, axis=None):
    return np.squeeze(input_._data, axis=axis)


@_wrap_ret()
def stack(inputs, axis=0):
    return np.stack(inputs, axis)


@_wrap_ret()
@_require_dense(0)
def std(input_, axis=None, keepdim=False):
    return np.std(input_._data, axis=axis, keepdims=keepdim)


@_wrap_ret()
def sum(input_, axis=None, dtype=None, keepdim=False):
    return np.sum(input_._data, axis=axis, dtype=dtype, keepdims=keepdim)


@_wrap_ret()
@_require_dense(0)
def take(input_, indices):
    return np.take(input_._data, indices)


@_wrap_ret()
def tan(input_):
    return np.tan(input_._data)


@_wrap_ret()
def tanh(input_):
    return np.tanh(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def tensordot(input_, other, axes=2):
    return np.tensordot(input_._data, other._data, axes)


@_wrap_ret()
@_require_dense(0)
def trace(input_):
    return np.trace(input_._data)


@_wrap_ret()
def transpose(input_, axes=None):
    return np.transpose(input_._data, axes)


@_wrap_ret()
@_require_dense(0)
def tril(input_, k=0):
    return np.tril(input_._data, k)


@_wrap_ret()
def tril_indices(n, m=0, offset=0):
    ret = np.tril_indices(n, k=offset, m=m)
    return tuple([DTensor(index) for index in ret])


@_require_dense(0)
def triu(input_, k=0):
    return np.triu(input_._data, k)


@_wrap_ret()
def triu_indices(n, m=0, offset=0):
    ret = np.triu_indices(n, k=offset, m=m)
    return tuple([DTensor(index) for index in ret])


@_wrap_ret()
def true_divide(input_, other):
    return np.true_divide(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def trunc(input_):
    return np.trunc(input_._data)


@_wrap_ret()
@_require_dense(0)
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    return np.unique(input_._data, return_inverse=return_inverse, return_counts=return_counts, axis=axis)


@_wrap_ret()
@_require_dense(0)
def var(input_, axis=None, keepdim=False):
    return np.var(input_._data, axis, keepdims=keepdim)


@_require_dense(1, 2)
def where(condition, x, y):
    return np.where(condition, x, y)


@_wrap_ret()
def zeros(shape, dtype=None):
    return np.zeros(shape, dtype=dtype)


@_require_dense(0)
def zeros_like(input_, dtype=None):
    return np.zeros_like(input_._data, dtype=dtype)
