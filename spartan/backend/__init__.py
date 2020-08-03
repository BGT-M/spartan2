import importlib
import json
import os
import sys
import warnings

from .cpu_backend import STensor, DTensor

__all__ = [
    'get_backend', 'get_preferred_backend', 'load_backend',
    'set_default_backend', 'STensor', 'DTensor'
]

# Inconsistent apis: sort(), round()
_APIS = [
    'add', 'all', 'angle', 'any', 'arange', 'argmax', 'argmin', 'argsort',
    'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
    'can_cast', 'ceil', 'complex128', 'complex64', 'conj', 'concatenate',
    'cos', 'cosh', 'cumprod', 'cumsum', 'diag', 'diagflat', 'diagonal',
    'dot', 'empty', 'empty_like', 'equal', 'exp', 'expm1', 'eye', 'flip',
    'float16', 'float32', 'float64', 'floor', 'floor_divide', 'fmod', 'full',
    'full_like', 'imag', 'int16', 'int32', 'int64', 'int8', 'isfinite',
    'isinf', 'isnan', 'linspace', 'log', 'log10', 'log1p', 'log2',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace',
    'matmul', 'mean', 'median', 'meshgrid', 'nonzero', 'ones', 'ones_like',
    'prod', 'real', 'reciprocal', 'remainder', 'reshape', 'roll', 'rot90',
    'short', 'sign', 'sin', 'sinh', 'split', 'sqrt', 'square', 'squeeze',
    'stack', 'std', 'sum', 'take', 'tan', 'tanh', 'tensordot', 'trace',
    'transpose', 'tril', 'tril_indices', 'triu', 'triu_indices', 'true_divide',
    'trunc', 'uint8', 'unique', 'var', 'where', 'zeros', 'zeros_like',
    'STensor', 'DTensor'
]

_backend_dict = {
    'cpu': 'cpu_backend',
    'gpu': 'gpu_backend'
}


_BACKEND = 'cpu'


def load_backend(backend_name):
    """Load specific backend for spartan.

    Parameters
    ----------
    backend_name : str, {'cpu', 'gpu'}
        Backend name
    """
    global _BACKEND
    _BACKEND = backend_name.lower()
    if _BACKEND not in _backend_dict:
        msg = f"Unsupported backend: {_BACKEND}. Use CPU backend instead."
        warnings.warn(msg)
        _BACKEND = 'cpu'
    # Check GPU
    if _BACKEND == 'gpu':
        mod = importlib.import_module('torch')
        if not mod.cuda.is_available():
            msg = "GPU not available! Use CPU backend instead."
            warnings.warn(msg)
            _BACKEND = 'cpu'

    print(f"Using backend {_BACKEND}")
    thismod = sys.modules['spartan']
    mod_name = '.' + _backend_dict[_BACKEND]
    mod = importlib.import_module(mod_name, __name__)
    for api in _APIS:
        if api in mod.__dict__:
            setattr(thismod, api, mod.__dict__[api])
        else:
            msg = f"'{api}' not implemented in '{_BACKEND}'"
            warnings.warn(msg)


def get_backend():
    """Get backend of spartan.

    Returns
    -------
    str
        Backend name
    """
    return _BACKEND


def set_default_backend(backend_name):
    """Save default backend setting to config file.

    Parameters
    ----------
    backend_name : str, {'cpu', 'gpu'}
        Default backend name
    """
    default_dir = os.path.join(os.path.expanduser('~'), '.spartan')
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, 'config.json')
    if os.path.exists(config_path):
        configs = json.load(open(config_path, 'r'))
        configs['backend'] = backend_name.lower()
        json.dump(configs, open(config_path, 'w'))
    print(f'Setting the default backend to {backend_name}.')
    print(f'The config is saved at ~/.spartan/config.json.')
    print(f'Environment variable `SPARTAN_BACKEND` is prefered.')


def get_preferred_backend():
    """Get preferred backend from environment variable or config file.

    Returns
    -------
    str
        Backend name.
    """
    config_path = os.path.join(
        os.path.expanduser('~'), '.spartan', 'config.json')
    backend_name = 'cpu'
    if "SPARTAN_CONFIG" in os.environ:
        backend_name = os.getenv('SPARTAN_BACKEND')
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get('backend', '').lower()

    backend_name = backend_name.lower()
    if backend_name in _backend_dict:
        return backend_name
    else:
        print(
            f"Spartan doesn't support backend {backend_name} yet. \
                Use cpu backend instead.")
        set_default_backend('cpu')
        return 'cpu'


load_backend(get_preferred_backend())
