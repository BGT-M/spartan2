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
    pass


@_wrap_ret
@_dispatch(np.all, sparse.COO.all)
def all(input_, axis=None, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.angle)
def angle(input_, deg=False):
    pass


@_wrap_ret
@_dispatch(np.any, sparse.COO.any)
def any(input_, axis=None, keepdims=False):
    pass


@_wrap_ret
def arange(start, stop, step, dtype=None):
    return np.arange(start, stop, step, dtype=dtype)


@_wrap_ret
@_dispatch(np.argmax)
def argmax(input_, axis=None):
    pass


@_wrap_ret
@_dispatch(np.argmin)
def argmin(input_, axis=None):
    pass


@_wrap_ret
@_dispatch(np.argsort)
def argsort(input_, axis=-1):
    pass


@_wrap_ret
@_dispatch(np.bincount)
def bincount(input_, weights, minlength=0):
    pass


@_wrap_ret
@_dispatch(np.bitwise_and)
def bitwise_and(input_, other):
    pass


@_wrap_ret
@_dispatch(np.bitwise_not)
def bitwise_not(input_, other):
    pass


@_wrap_ret
@_dispatch(np.bitwise_or)
def bitwise_or(input_, other):
    pass


@_wrap_ret
@_dispatch(np.bitwise_xor)
def bitwise_xor(input_, other):
    pass


@_dispatch(np.can_cast)
def can_cast(from_, to):
    pass


@_wrap_ret
@_dispatch(np.ceil)
def ceil(input_):
    pass


@_wrap_ret
@_dispatch(np.conj, sparse.COO.conj)
def conj(input_):
    pass


@_wrap_ret
@_dispatch(np.cos)
def cos(input_):
    pass


@_wrap_ret
@_dispatch(np.cosh)
def cosh(input_):
    pass


@_wrap_ret
@_dispatch(np.cross)
def cross(input_, other, axis=-1):
    pass


@_wrap_ret
@_dispatch(np.cumprod)
def cumprod(input_, axis=None, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.cumsum)
def cumsum(input_, axis=None, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.diag)
def diag(input_, k):
    pass


@_wrap_ret
@_dispatch(np.diagflat)
def diagflat(input_, offset=0):
    pass


@_wrap_ret
@_dispatch(np.diagonal, sparse.diagonal)
def diagonal(input_, offset=0, axis1=0, axis2=1):
    pass


@_wrap_ret
@_dispatch(np.dot, sparse.dot)
def dot(input_, other):
    pass


@_wrap_ret
def einsum(equation, *operands):
    return np.einsum(equation, *operands)


@_wrap_ret
@_dispatch(np.empty)
def empty(shape, dtype):
    pass


@_wrap_ret
@_dispatch(np.empty_like)
def empty_like(input_, dtype):
    pass


@_wrap_ret
@_dispatch(np.equal)
def equal(input_, other):
    pass


@_wrap_ret
@_dispatch(np.exp)
def exp(input_):
    pass


@_wrap_ret
@_dispatch(np.expm1, np.expm1)
def expm1(input_):
    pass


@_wrap_ret
@_dispatch(np.eye, sparse.eye)
def eye(n, m=None, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.flip)
def flip(input_, axis=None):
    pass


@_wrap_ret
@_dispatch(np.floor)
def floor(input_):
    pass


@_wrap_ret
@_dispatch(np.floor_divide)
def floor_divide(input_, other):
    pass


@_wrap_ret
@_dispatch(np.fmod)
def fmod(input_, other):
    pass


@_wrap_ret
@_dispatch(np.full, sparse.full)
def full(shape, value, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.full_like, sparse.full_like)
def full_like(input_, value, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.imag)
def imag(input_):
    pass


@_wrap_ret
@_dispatch(np.isfinite)
def isfinite(input_):
    pass


@_wrap_ret
@_dispatch(np.isinf)
def isinf(input_):
    pass


@_wrap_ret
@_dispatch(np.isnan)
def isnan(input_):
    pass


@_wrap_ret
@_dispatch(np.linspace)
def linspace(start, end, step, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.log)
def log(input_):
    pass


@_wrap_ret
@_dispatch(np.log10)
def log10(input_):
    pass


@_wrap_ret
@_dispatch(np.log1p, np.log1p)
def log1p(input_):
    pass


@_wrap_ret
@_dispatch(np.log2)
def log2(input_):
    pass


@_wrap_ret
@_dispatch(np.logical_and)
def logical_and(input_, other):
    pass


@_wrap_ret
@_dispatch(np.logical_not)
def logical_not(input_):
    pass


@_wrap_ret
@_dispatch(np.logical_or)
def logical_or(input_, other):
    pass


@_wrap_ret
@_dispatch(np.logical_xor)
def logical_xor(input_, other):
    pass


@_wrap_ret
@_dispatch(np.logspace)
def logspace(start, stop, step, base=10, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.matmul, sparse.matmul)
def matmul(input_, other):
    pass


@_wrap_ret
@_dispatch(np.mean, sparse.COO.mean)
def mean(input_, axis=None, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.median)
def median(input_, axis=-1, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.meshgrid)
def meshgrid(*inputs):
    pass


@_wrap_ret
@_dispatch(np.nonzero, sparse.COO.nonzero)
def nonzero(input_):
    pass


@_wrap_ret
@_dispatch(np.ones, sparse.ones)
def ones(shape, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.ones_like, sparse.ones_like)
def ones_like(input_, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.prod, sparse.COO.prod)
def prod(input_, axis=None, keepdims=False, dtype=None):
    pass


@_wrap_ret
@_dispatch(np.real)
def real(input_):
    pass


@_wrap_ret
@_dispatch(np.reciprocal)
def reciprocal(input_):
    pass


@_wrap_ret
@_dispatch(np.remainder)
def remainder(input_, other):
    pass


@_wrap_ret
@_dispatch(np.reshape, sparse.COO.reshape)
def reshape(input_, shape):
    pass


@_wrap_ret
@_dispatch(np.roll)
def roll(input_, shift, axis=None):
    pass


@_wrap_ret
@_dispatch(np.rot90)
def rot90(input_, k=1, axes=(0, 1)):
    pass


@_wrap_ret
@_dispatch(np.sign)
def sign(input_):
    pass


@_wrap_ret
@_dispatch(np.sin, np.sin)
def sin(input_):
    pass


@_wrap_ret
@_dispatch(np.sinh, np.sinh)
def sinh(input_):
    pass


@_wrap_ret
@_dispatch(np.split)
def split(input_, indices_or_sections, axis=0):
    pass


@_wrap_ret
@_dispatch(np.sqrt, np.sqrt)
def sqrt(input_):
    pass


@_wrap_ret
@_dispatch(np.square, np.square)
def square(input_):
    pass


@_wrap_ret
@_dispatch(np.squeeze)
def squeeze(input_, axis=None):
    pass


@_wrap_ret
@_dispatch(np.stack, sparse.stack)
def stack(inputs, axis=0):
    pass


@_wrap_ret
@_dispatch(np.std, sparse.COO.std)
def std(input_, axis=None, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.sum, sparse.COO.sum)
def sum(input_, axis=None, dtype=None, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.take)
def take(input_, indices):
    pass


@_wrap_ret
@_dispatch(np.tan, np.tan)
def tan(input_):
    pass


@_wrap_ret
@_dispatch(np.tanh, np.tanh)
def tanh(input_):
    pass


@_wrap_ret
@_dispatch(np.tensordot, sparse.tensordot)
def tensordot(input_, other, axes=2):
    pass


@_wrap_ret
@_dispatch(np.trace, np.trace)
def trace(input_):
    pass


@_wrap_ret
@_dispatch(np.transpose, sparse.COO.transpose)
def transpose(input_, axes=None):
    pass


@_wrap_ret
@_dispatch(np.tril, sparse.tril)
def tril(input_, k=0):
    pass


@_wrap_ret
@_dispatch(np.tril_indices)
def tril_indices(n, m=0, offset=0):
    pass


@_wrap_ret
@_dispatch(np.triu, sparse.triu)
def triu(input_, k=0):
    pass


@_wrap_ret
@_dispatch(np.triu_indices)
def triu_indices(n, m=0, offset=0):
    pass


@_wrap_ret
@_dispatch(np.true_divide)
def true_divide(input_, other):
    pass


@_wrap_ret
@_dispatch(np.trunc)
def trunc(input_):
    pass


@_wrap_ret
@_dispatch(np.unique)
def unique(input_, return_inverse=False, return_counts=False, axis=None):
    pass


@_wrap_ret
@_dispatch(np.var, sparse.COO.var)
def var(input_, axis=None, keepdims=False):
    pass


@_wrap_ret
@_dispatch(np.where, sparse.where)
def where(condition, x, y):
    pass


@_wrap_ret
@_dispatch(np.zeros, sparse.zeros)
def zeros(shape, dtype=None):
    return np.zeros(shape, dtype=dtype)


@_wrap_ret
@_dispatch(np.zeros_like, sparse.zeros_like)
def zeros_like(input_, dtype=None):
    pass
