Process of adding a new model.

1. Add source code to spartan2/spartan/model/

2. Insert specific function into model file.

E.g. In `./beatlex/Beatlex.py`

``` python
class Beatlex(DMmodel):
    def __init__(self):
        pass

    # design for summarization task
    def summarization(self, params=None):
        pass

    # design for DMmodel
    def run(self):
        pass
```

3. Add call function for dynamic import.

E.g. In `./beatlex/__init__.py`

``` python
# Import BeatLex class from file 
from .Beatlex import BeatLex

def __call__():
    return Beatlex
```

4. Register model path to `Task.py` file:

E.g. Beatlex realizes code for summarization task.

In `../task/summarization.py`

``` python
# MODEL_PATH is a global variabel which is defined in 'spartan.model
class SumPolicy(Enum):
    '''Registration for path of models who can do summarization task.
    '''
    # Only need path of file, not class
    Beatlex = MODEL_PATH + ".beatlex"
```

5. Register model path to `../model/__init__.py` file:

E.g. Beatlex realizes code for model.

In `../model/__init__.py`

``` python
# Design a model_name: BeatLex
# Subsititute the path of model: ".beatlex"
# Keep others [partial, __call__, MODEL_PATH] unchanged
BeatLex = partial(__call__, MODEL_PATH + ".beatlex")
Holoscope = partial(__call__, MODEL_PATH + ".holoscope")
```

6. Add `model_name` to `__all__` in `../model/__init__.py`

``` python
__all__ = [
    'BeatLex',
]
```
