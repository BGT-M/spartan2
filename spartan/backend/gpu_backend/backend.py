import math
import torch
from .tensor import STensor, DTensor

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


def _wrap_ret(ret):
    if isinstance(ret, torch.Tensor):
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


def _check_single(input_, func):
    if isinstance(input_, STensor):
        return STensor
    elif isinstance(input_, DTensor):
        return DTensor
    else:
        raise TypeError(
            f"`st.{func.__name__}` does not support {type(input_)} type.")


def _check_double(input_, other, func):
    if isinstance(input_, STensor) and isinstance(other, STensor):
        return STensor
    elif isinstance(input_, DTensor) and isinstance(other, DTensor):
        return DTensor
    else:
        raise TypeError(
            f"`st.{func.__name__}` does not support {type(input_)} and {type(other)} type.")


def add(input_, other):
    # type_ = _check_double(input_, other, add)
    return type_(torch.add(input_.data, other.data))


def all(input_, dim=None, keepdim=False):
    _check_dense(input_, all)
    ret = torch.all(input_.data, dim=dim, keepdim=keepdim)
    return _wrap_ret(ret)


def allclose(input_, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    _check_dense(input_, allclose, other)
    return torch.allclose(input_.data, other.data, rtol, atol, equal_nan)


def angle(input_, deg=False):
    _check_dense(input_, angle)
    if deg:
        ret = torch.angle(input_) * 180 / math.pi
    else:
        ret = torch.angle(input_)
    return _wrap_ret(ret)


def any(input_, dim, keepdim=False):
    _check_dense(input_, any)
    ret = torch.any(input_.data, dim, keepdim)
    return _wrap_ret(ret)


def arange(start, stop, step, dtype=None):
    return DTensor(torch.arange(start, stop, step, dtype=dtype))


def argmax(input_, dim=None) -> DTensor:
    type_ = _check_single(input_, argmax)
    if dim is None:
        ret = torch.argmin(input_)
    else:
        ret = torch.argmax(input_, dim=dim)
    # if type_ == STensor:
    #     ret = torch.tensor(ret).flatten()
    return DTensor(ret)


def argmin(input_, dim=None) -> DTensor:
    if dim is None:
        ret = torch.argmin(input_)
    else:
        ret = torch.argmax(input_, dim=dim)
    # if type_ == STensor:
    #     ret = torch.tensor(ret).flatten()
    return DTensor(ret)


def argsort(input_: DTensor, dim=-1) -> DTensor:
    _check_dense(input_, argsort)
    return DTensor(torch.argsort(input_.data, dim=dim))


def bincount(input_: DTensor, weights, minlength=0) -> DTensor:
    _check_dense(input_, bincount)
    return DTensor(torch.bincount(input_.data, weights, minlength))


def bitwise_and(input_: DTensor, other: DTensor) -> DTensor:
    _check_dense(input_, bitwise_and, other)
    ret = torch.bitwise_and(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_not(input_, other):
    _check_dense(input_, bitwise_and, other)
    ret = torch.bitwise_not(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_or(input_, other):
    _check_dense(input_, bitwise_or, other)
    ret = torch.bitwise_or(input_.data, other.data)
    return _wrap_ret(ret)


def bitwise_xor(input_, other):
    _check_dense(input_, bitwise_xor, other)
    ret = torch.bitwise_xor(input_.data, other.data)
    return _wrap_ret(ret)


def can_cast(from_, to):
    return torch.can_cast(from_, to)


def ceil(input_):
    _check_dense(input_, ceil)
    return DTensor(torch.ceil(input_.data))


def conj(input_):
    type_ = _check_single(input_, conj)
    return type_(torch.conj(input_.data))


def cos(input_):
    _check_dense(input_, cos)
    return DTensor(torch.cos(input_.data))


def cosh(input_):
    _check_dense(input_, cosh)
    return DTensor(torch.cos(input_.data))


def cross(input_, other, dim=-1):
    _check_dense(input_, cross, other)
    return DTensor(torch.cross(input_.data, other.data, dim=dim))


def cumprod(input_, dim=None, dtype=None):
    _check_dense(input_, cumprod)
    return DTensor(torch.cumprod(input_.data, dim=dim, dtype=dtype))


def cumsum(input_, dim=None, dtype=None):
    _check_dense(input_, cumsum)
    return DTensor(torch.cumsum(input_.data, dim, dtype))


def diag(input_, k):
    _check_dense(input_, diag)
    return DTensor(torch.diag(input_.data, k))


def diagflat(input_, offset=0):
    _check_dense(input_, diagflat)
    return DTensor(torch.diagflat(input_.data, offset))


def diagonal(input_, offset, dim1=None, dim2=None):
    _check_dense(input_, diagonal)
    return DTensor(torch.diagonal(input_.data, offset, dim1, dim2))


def dot(input_, other):
    type_ = _check_double(input_, other, bitwise_and)
    return type_(torch.dot(input_.data, other.data))


def einsum(equation, *operands):
    return DTensor(torch.eigsum(equation, *operands))


def empty(shape, dtype):
    return DTensor(torch.empty(shape, dtype=dtype))


def empty_like(input_, dtype):
    _check_dense(input_, empty_like)
    return DTensor(torch.empty_like(input_.data, dtype=dtype))


def equal(input_, other):
    _check_dense(input_, equal, other)
    return DTensor(torch.equal(input_.data, other.data))


def exp(input_):
    _check_dense(input_, exp)
    return DTensor(torch.exp(input_.data))


def expm1(input_):
    type_ = _check_single(input_, expm1)
    return type_(torch.expm1(input_.data))


def eye(n, m=None, dtype=None):
    return DTensor(torch.eye(n, m, dtype=dtype))


def flip(input_, dim=None):
    type_ = _check_single(input_, flip)
    return type_(torch.flip(input_.data, dim))


def floor(input_):
    _check_dense(input_, floor)
    return DTensor(torch.floor(input_.data))


def floor_divide(input_, other):
    _check_dense(input_, floor_divide, other)
    return DTensor(torch.floor_divide(input_.data, other.data))


def fmod(input_, other):
    _check_dense(input_, fmod, other)
    return DTensor(torch.fmod(input_.data, other.data))


def full(shape, value, dtype=None):
    return DTensor(torch.full(shape, value, dtype=dtype))


def full_like(input_, value, dtype=None):
    _check_dense(input_, full_like)
    return DTensor(torch.full_like(input_.data, value, dtype=dtype))


def imag(input_):
    type_ = _check_single(input_, imag)
    return type_(torch.imag(input_.data))


def isfinite(input_):
    _check_dense(input_, isfinite)
    return DTensor(torch.isfinite(input_.data))


def isinf(input_):
    _check_dense(input_, isinf)
    return DTensor(torch.isinf(input_.data))


def isnan(input_):
    _check_dense(input_, isnan)
    return DTensor(torch.isnan(input_.data))


def linspace(start, end, step, dtype=None):
    return DTensor(torch.linspace(start, end, step, dtype=dtype))


def log(input_):
    _check_dense(input_, log)
    return DTensor(torch.log(input_.data))


def log10(input_):
    _check_dense(input_, log10)
    return DTensor(torch.log10(input_.data))


def log1p(input_):
    _check_dense(input_, log1p)
    return DTensor(torch.log1p(input_.data))


def log2(input_):
    _check_dense(input_, log2)
    return DTensor(torch.log2(input_.data))


def logical_and(input_, other):
    _check_dense(input_, logical_and, other)
    return DTensor(torch.logical_and(input_.data, other.data))


def logical_not(input_):
    _check_dense(input_, logical_not)
    return DTensor(torch.logical_not(input_.data))


def logical_or(input_, other):
    _check_dense(input_, logical_or, other)
    return DTensor(torch.logical_or(input_.data, other.data))


def logical_xor(input_, other):
    _check_dense(input_, logical_xor, other)
    return DTensor(torch.logical_xor(input_.data, other.data))


def logspace(start, stop, step, base=10, dtype=None):
    return DTensor(torch.logspace(start, stop, step, base=base, dtype=dtype))


def matmul(input_, other):
    _check_dense(input_, matmul, other)
    return DTensor(torch.matmul(input_.data, other.data))


def mean(input_, dim=None, keepdim=False):
    type_ = _check_single(input_, mean)
    if dim is None:
        ret = torch.mean(input_.data)
    else:
        ret = torch.mean(input_.data, dim=dim, keepdim=keepdim)
    # if type_ == STensor:
    #     ret = torch.array(ret).flatten()
    return DTensor(ret)


def median(input_, dim=-1, keepdim=False):
    type_ = _check_single(input_, median)
    if dim is None:
        ret = torch.median(input_.data)
    else:
        ret = torch.median(input_.data, dim=dim, keepdim=keepdim)
    # if type_ == STensor:
    #     ret = torch.tensor(ret).flatten()
    return DTensor(ret)


def meshgrid(*inputs):
    datas = [i.data for i in inputs]
    return tuple([DTensor(d) for d in torch.meshgrid(*datas)])


def nonzero(input_):
    return tuple([DTensor(d) for d in torch.nonzero(input_.data, as_tuple=True)])


def ones(shape, dtype=None):
    return DTensor(torch.ones(shape, dtype=dtype))


def ones_like(input_, dtype=None):
    _check_dense(input_, ones_like)
    return DTensor(torch.ones_like(input_.data, dtype=dtype))


def prod(input_, dim=None, keepdim=False, dtype=None):
    _check_dense(input_, prod)
    ret = torch.prod(input_.data, dim=dim, keepdim=keepdim, dtype=dtype)
    return _wrap_ret(ret)


def real(input_):
    type_ = _check_single(input_, real)
    return type_(torch.real(input_.data))


def reciprocal(input_):
    _check_dense(input_, reciprocal)
    return DTensor(torch.reciprocal(input_.data))


def remainder(input_, other):
    _check_dense(input_, remainder, other)
    return DTensor(torch.remainder(input_.data, other.data))


def reshape(input_, shape):
    type_ = _check_single(input_, reshape)
    return type_(torch.rehsape(input_.data, shape))


def roll(input_, shift, dim=None):
    _check_dense(input_, roll)
    return DTensor(torch.roll(input_.data, shift, dims=dim))


def rot90(input_, k=1, axes=(0, 1)):
    _check_dense(innput_, rot90)
    return DTensor(torch.rot90(input_.data, k, dims=axes))


def round(input_):
    type_ = _check_single(input_, round)
    return type_(torch.round(input_.data))


def sign(input_):
    _check_dense(input_, sign)
    return DTensor(torch.sign(input_.data))


def sin(input_):
    type_ = _check_single(input_, sin)
    return type_(torch.sin(input_.data))


def sinh(input_):
    type_ = _check_single(input_, sinh)
    return type_(torch.sinh(input_.data))


def sort(input_, dim=-1):
    _check_dense(input_, sort)
    return DTensor(torch.sort(input_.data, dim=dim))


def split(input_, indices_or_sections, dim=0):
    _check_dense(input_, split)
    return DTensor(torch.split(input_.data, indices_or_sections, dim))


def sqrt(input_):
    type_ = _check_single(input_, sqrt)
    return type_(torch.sqrt(input_.data))


def square(input_):
    type_ = _check_single(input_, square)
    return type_(torch.square(input_.data))


def squeeze(input_, dim=None):
    _check_dense(input_, squeeze)
    return DTensor(torch.squeeze(input_.data, dim=dim))


def stack(inputs, dim=0):
    return DTensor(torch.stack(inputs, dim))


def std(input_, dim=None, keepdim=False):
    _check_dense(input_, std)
    if dim is None:
        ret = torch.std(input_.data, unbiased=False)
    else:
        ret = torch.std(input_.data, dim=dim, keepdim=keepdim, unbiased=False)
    return DTensor(ret)


def sum(input_, dim=None, dtype=None, keepdim=False):
    if dim is None:
        ret = torch.sum(input_.data, dtype=dtype)
    else:
        ret = torch.sum(input_.data, dim=dim, dtype=dtype, keepdim=keepdim)
    return _wrap_ret(ret)


def take(input_, indices):
    _check_dense(input_, take)
    return DTensor(torch.take(input_.data, indices))


def tan(input_):
    type_ = _check_single(input_, tan)
    return type_(torch.tan(input_.data))


def tanh(input_):
    type_ = _check_single(input_, tan)
    return type_(torch.tanh(input_.data))


def tensordot(input_, other, axes=2):
    _check_dense(input_, tensordot, other)
    return DTensor(torch.tensordot(input_.data, other.data, axes))


def trace(input_):
    _check_dense(input_)
    ret = torch.trace(input_.data)
    return _wrap_ret(ret)


def transpose(input_, axes=None):
    type_ = _check_single(input_, transpose)
    if axes is None:
        axes = (0, 1)
    return type_(torch.transpose(input_, axes[0], axes[1]))


def tril(input_, k=0):
    _check_dense(input_, tril)
    return DTensor(torch.tril(input_.data, k))


def tril_indices(n, m=0, offset=0):
    ret = torch.tril_indices(row=m, col=m, offset=offset)
    return tuple([DTensor(index) for index in ret])


def triu(input_, k=0):
    _check_dense(input_, triu)
    return DTensor(torch.triu(input_.data, k))


def triu_indices(n, m=0, offset=0):
    ret = torch.triu_indices(row=m, col=m, offset=offset)
    return tuple([DTensor(index) for index in ret])


def true_divide(input_, other):
    return DTensor(torch.true_divide(input_.data, other.data))


def trunc(input_):
    _check_dense(input_, trunc)
    return DTensor(torch.trunc(input_.data))


def unique(input_, return_inverse=False, return_counts=False, dim=None):
    _check_dense(input_, unique)
    return DTensor(torch.unique(input_.data, return_inverse=return_inverse, return_counts=return_counts, dim=dim))


def var(input_, dim=None, keepdim=False):
    _check_dense(input_, var)
    if axis is None:
        ret = torch.var(input_)
    else:
        ret = torch.var(input_, dim=dim, keepdim=keepdim)
    return _wrap_ret(ret)


def where(condition, x, y):
    _check_dense(x, where, y)
    return DTensor(torch.where(condition, x, y))


def zeros(shape, dtype=None):
    return DTensor(torch.zeros(shape, dtype=dtype))


def zeros_like(input_, dtype=None):
    _check_dense(input_, zeros_like)
    return DTensor(torch.zeros_like(input_.data, dtype=dtype))
