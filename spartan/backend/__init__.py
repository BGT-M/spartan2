import importlib
import json
import os
import sys

_BACKEND = ''

_APIS = [
    'add', 'all', 'allclose', 'angle', 'any', 'arange', 'argmax', 'argmin',
    'argsort', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or',
    'bitwise_xor', 'can_cast', 'ceil', 'complex128', 'complex64', 'conj',
    'cos', 'cosh', 'cross', 'cumprod', 'cumsum', 'diag', 'diagflat',
    'diagonal', 'dot', 'double', 'dtype', 'einsum', 'empty', 'empty_like',
    'equal', 'exp', 'expm1', 'eye', 'fft', 'finfo', 'flip', 'float16',
    'float32', 'float64', 'floor', 'floor_divide', 'fmod', 'full',
    'full_like', 'half', 'iinfo', 'imag', 'int16', 'int32', 'int64',
    'int8', 'isclose', 'isfinite', 'isinf', 'isnan', 'linspace',
    'log', 'log10', 'log1p', 'log2', 'logical_and', 'logical_not',
    'logical_or', 'logical_xor', 'logspace', 'matmul', 'mean', 'median',
    'meshgrid', 'nonzero', 'ones', 'ones_like', 'prod', 'promote_types',
    'random', 'real', 'reciprocal', 'remainder', 'reshape', 'result_type',
    'roll', 'rot90', 'round_', 'save', 'select', 'set_printoptions',
    'short', 'sign', 'sin', 'sinh', 'sort', 'split', 'sqrt', 'square',
    'squeeze', 'stack', 'std', 'sum', 'take', 'tan', 'tanh', 'tensordot',
    'trace', 'transpose', 'trapz', 'tril', 'tril_indices', 'triu',
    'triu_indices', 'true_divide', 'trunc', 'typename', 'uint8',
    'unique', 'var', 'where', 'zeros', 'zeros_like'
]


def load_backend(backend_name):
    global _BACKEND
    _BACKEND = backend_name
    print(f"Using backend {backend_name}")
    module_name = backend_name
    if backend_name == 'pytorch':
        module_name = 'torch'
    mod = importlib.import_module(module_name)
    thismod = sys.modules['spartan']

    for api in _APIS:
        if api in mod.__dict__:
            setattr(thismod, api, mod.__dict__[api])
        else:
            print(f'Warning: API `{api}` not in module: `{backend_name}`')


def backend():
    return _BACKEND


def set_default_backend(backend_name):
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
    config_path = os.path.join(
        os.path.expanduser('~'), '.spartan', 'config.json')
    backend_name = 'numpy'
    if "SPARTAN_CONFIG" in os.environ:
        backend_name = os.getenv('SPARTAN_BACKEND')
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get('backend', '').lower()

    if (backend_name in ['numpy', 'scipy', 'pytorch']):
        return backend_name
    else:
        print(
            f"Spartan doesn't support backend {backend_name} yet. Using numpy instead.")
        set_default_backend('numpy')
        return 'numpy'


load_backend(get_preferred_backend())
