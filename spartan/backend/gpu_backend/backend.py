import math
import torch
from .tensor import STensor, DTensor, _check_params, _require_dense, _wrap_ret

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


@_wrap_ret()
@_check_params(0, 1)
def add(input_, other):
    return torch.add(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def all(input_, axis=None, keepdim=False):
    return torch.all(input_._data, dim=axis, keepdim=keepdim)


@_wrap_ret()
@_require_dense(0, 1)
def allclose(input_, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    return torch.allclose(input_._data, other._data, rtol, atol, equal_nan)


@_wrap_ret()
@_require_dense(0)
def angle(input_, deg=False):
    if deg:
        ret = torch.angle(input_) * 180 / math.pi
    else:
        ret = torch.angle(input_)
    return ret


@_wrap_ret()
@_require_dense(0)
def any(input_, axis, keepdim=False):
    return torch.any(input_._data, axis, keepdim)


@_wrap_ret()
def arange(start, stop, step, dtype=None):
    return torch.arange(start, stop, step, dtype=dtype)


@_wrap_ret()
@_check_params(0)
def argmax(input_, axis=None):
    if axis is None:
        return torch.argmin(input_)
    else:
        return torch.argmax(input_, dim=axis)


@_wrap_ret()
@_check_params(0)
def argmin(input_, axis=None) -> DTensor:
    if axis is None:
        return torch.argmin(input_)
    else:
        return torch.argmax(input_, dim=axis)


@_wrap_ret()
@_require_dense(0)
def argsort(input_: DTensor, axis=-1) -> DTensor:
    return torch.argsort(input_._data, dim=axis)


@_wrap_ret()
@_require_dense(0)
def bincount(input_: DTensor, weights, minlength=0) -> DTensor:
    return torch.bincount(input_._data, weights, minlength)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_and(input_: DTensor, other: DTensor) -> DTensor:
    return torch.bitwise_and(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_not(input_, other):
    return torch.bitwise_not(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_or(input_, other):
    return torch.bitwise_or(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def bitwise_xor(input_, other):
    return torch.bitwise_xor(input_._data, other._data)


def can_cast(from_, to):
    return torch.can_cast(from_, to)


@_wrap_ret()
@_require_dense(0)
def ceil(input_):
    return torch.ceil(input_._data)


@_wrap_ret()
def conj(input_):
    return torch.conj(input_._data)


@_wrap_ret()
@_require_dense(0)
def cos(input_):
    return torch.cos(input_._data)


@_wrap_ret()
@_require_dense(0)
def cosh(input_):
    return torch.cos(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def cross(input_, other, axis=-1):
    return torch.cross(input_._data, other._data, dim=axis)


@_wrap_ret()
@_require_dense(0)
def cumprod(input_, axis=None, dtype=None):
    return torch.cumprod(input_._data, dim=axis, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def cumsum(input_, axis=None, dtype=None):
    return torch.cumsum(input_._data, axis, dtype)


@_wrap_ret()
@_require_dense(0)
def diag(input_, k):
    return torch.diag(input_._data, k)


@_wrap_ret()
@_require_dense(0)
def diagflat(input_, offset=0):
    return torch.diagflat(input_._data, offset)


@_wrap_ret()
@_require_dense(0)
def diagonal(input_, offset, axis1=None, axis2=None):
    return torch.diagonal(input_._data, offset, axis1, axis2)


@_wrap_ret()
@_check_params(0, 1)
def dot(input_, other):
    return torch.dot(input_._data, other._data)


@_wrap_ret()
def einsum(equation, *operands):
    return torch.eigsum(equation, *operands)


@_wrap_ret()
def empty(shape, dtype):
    return torch.empty(shape, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def empty_like(input_, dtype):
    return torch.empty_like(input_._data, dtype=dtype)


@_wrap_ret()
@_require_dense(0, 1)
def equal(input_, other):
    return torch.equal(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def exp(input_):
    return torch.exp(input_._data)


@_wrap_ret()
def expm1(input_):
    return torch.expm1(input_._data)


@_wrap_ret()
def eye(n, m=None, dtype=None):
    return torch.eye(n, m, dtype=dtype)


@_wrap_ret()
@_check_params(0)
def flip(input_, axis=None):
    return torch.flip(input_._data, axis)


@_wrap_ret()
@_require_dense(0)
def floor(input_):
    return torch.floor(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def floor_divide(input_, other):
    return torch.floor_divide(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def fmod(input_, other):
    return torch.fmod(input_._data, other._data)


@_wrap_ret()
def full(shape, value, dtype=None):
    return torch.full(shape, value, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def full_like(input_, value, dtype=None):
    return torch.full_like(input_._data, value, dtype=dtype)


@_wrap_ret()
def imag(input_):
    return torch.imag(input_._data)


@_wrap_ret()
@_require_dense(0)
def isfinite(input_):
    return torch.isfinite(input_._data)


@_wrap_ret()
@_require_dense(0)
def isinf(input_):
    return torch.isinf(input_._data)


@_wrap_ret()
@_require_dense(0)
def isnan(input_):
    return torch.isnan(input_._data)


@_wrap_ret()
def linspace(start, end, step, dtype=None):
    return torch.linspace(start, end, step, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def log(input_):
    return torch.log(input_._data)


@_wrap_ret()
@_require_dense(0)
def log10(input_):
    return torch.log10(input_._data)


@_wrap_ret()
def log1p(input_):
    return torch.log1p(input_._data)


@_wrap_ret()
@_require_dense(0)
def log2(input_):
    return torch.log2(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_and(input_, other):
    return torch.logical_and(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def logical_not(input_):
    return torch.logical_not(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_or(input_, other):
    return torch.logical_or(input_._data, other._data)


@_wrap_ret()
@_require_dense(0, 1)
def logical_xor(input_, other):
    return torch.logical_xor(input_._data, other._data)


@_wrap_ret()
def logspace(start, stop, step, base=10, dtype=None):
    return torch.logspace(start, stop, step, base=base, dtype=dtype)


@_wrap_ret()
@_require_dense(0, 1)
def matmul(input_, other):
    return torch.matmul(input_._data, other._data)


@_wrap_ret()
def mean(input_, axis=None, keepdim=False):
    if axis is None:
        ret = torch.mean(input_._data)
    else:
        ret = torch.mean(input_._data, dim=axis, keepdim=keepdim)
    return ret


@_wrap_ret()
def median(input_, axis=-1, keepdim=False):
    if axis is None:
        ret = torch.median(input_._data)
    else:
        ret = torch.median(input_._data, dim=axis, keepdim=keepdim)
    return ret


@_wrap_ret()
def meshgrid(*inputs):
    datas = [i._data for i in inputs]
    return tuple([d for d in torch.meshgrid(*datas)])


@_wrap_ret()
def nonzero(input_):
    return tuple([d for d in torch.nonzero(input_._data, as_tuple=True)])


@_wrap_ret()
def ones(shape, dtype=None):
    return torch.ones(shape, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def ones_like(input_, dtype=None):
    return torch.ones_like(input_._data, dtype=dtype)


@_wrap_ret()
@_require_dense(0)
def prod(input_, axis=None, keepdim=False, dtype=None):
    return torch.prod(input_._data, dim=axis, keepdim=keepdim, dtype=dtype)


@_wrap_ret()
def real(input_):
    return torch.real(input_._data)


@_wrap_ret()
@_require_dense(0)
def reciprocal(input_):
    return torch.reciprocal(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def remainder(input_, other):
    return torch.remainder(input_._data, other._data)


@_wrap_ret()
def reshape(input_, shape):
    return torch.rehsape(input_._data, shape)


@_wrap_ret()
@_require_dense(0)
def roll(input_, shift, axis=None):
    return torch.roll(input_._data, shift, dims=axis)


@_wrap_ret()
@_require_dense(0)
def rot90(input_, k=1, axes=(0, 1)):
    return torch.rot90(input_._data, k, dims=axes)


@_wrap_ret()
@_require_dense(0)
def round(input_):
    return torch.round(input_._data)


@_wrap_ret()
def sign(input_):
    return torch.sign(input_._data)


@_wrap_ret()
def sin(input_):
    return torch.sin(input_._data)


@_wrap_ret()
def sinh(input_):
    return torch.sinh(input_._data)


@_wrap_ret()
@_require_dense(0)
def split(input_, indices_or_sections, axis=0):
    return torch.split(input_._data, indices_or_sections, axis)


@_wrap_ret()
def sqrt(input_):
    return torch.sqrt(input_._data)


@_wrap_ret()
def square(input_):
    return torch.square(input_._data)


@_wrap_ret()
@_require_dense(0)
def squeeze(input_, axis=None):
    return torch.squeeze(input_._data, dim=axis)


@_wrap_ret()
def stack(inputs, axis=0):
    return torch.stack(inputs, axis)


@_wrap_ret()
@_require_dense(0)
def std(input_, axis=None, keepdim=False):
    if axis is None:
        ret = torch.std(input_._data, unbiased=False)
    else:
        ret = torch.std(input_._data, dim=axis,
                        keepdim=keepdim, unbiased=False)
    return ret


@_wrap_ret()
def sum(input_, axis=None, dtype=None, keepdim=False):
    if axis is None:
        ret = torch.sum(input_._data, dtype=dtype)
    else:
        ret = torch.sum(input_._data, dim=axis, dtype=dtype, keepdim=keepdim)
    return ret


@_wrap_ret()
@_require_dense(0)
def take(input_, indices):
    return torch.take(input_._data, indices)


@_wrap_ret()
def tan(input_):
    return torch.tan(input_._data)


@_wrap_ret()
def tanh(input_):
    return torch.tanh(input_._data)


@_wrap_ret()
@_require_dense(0, 1)
def tensordot(input_, other, axes=2):
    return torch.tensordot(input_._data, other._data, axes)


@_wrap_ret()
@_require_dense(0)
def trace(input_):
    return torch.trace(input_._data)


@_wrap_ret()
def transpose(input_, axes=None):
    if axes is None:
        axes = (0, 1)
    return torch.transpose(input_, axes[0], axes[1])


@_wrap_ret()
@_require_dense(0)
def tril(input_, k=0):
    return torch.tril(input_._data, k)


@_wrap_ret()
def tril_indices(n, m=0, offset=0):
    return torch.tril_indices(row=m, col=m, offset=offset)


@_require_dense(0)
def triu(input_, k=0):
    return torch.triu(input_._data, k)


@_wrap_ret()
def triu_indices(n, m=0, offset=0):
    ret = torch.triu_indices(row=m, col=m, offset=offset)
    return tuple([index for index in ret])


@_wrap_ret()
def true_divide(input_, other):
    return torch.true_divide(input_._data, other._data)


@_wrap_ret()
@_require_dense(0)
def trunc(input_):
    return torch.trunc(input_._data)


@_wrap_ret()
@_require_dense(0)
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    return torch.unique(input_._data, return_inverse=return_inverse, return_counts=return_counts, dim=axis)


@_wrap_ret()
@_require_dense(0)
def var(input_, axis=None, keepdim=False):
    if axis is None:
        ret = torch.var(input_)
    else:
        ret = torch.var(input_, dim=axis, keepdim=keepdim)
    return ret


@_require_dense(1, 2)
def where(condition, x, y):
    return torch.where(condition, x, y)


@_wrap_ret()
def zeros(shape, dtype=None):
    return torch.zeros(shape, dtype=dtype)


@_require_dense(0)
def zeros_like(input_, dtype=None):
    return torch.zeros_like(input_._data, dtype=dtype)
