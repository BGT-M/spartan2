Process of adding a new model.

1. Add source code to spartan2/spartan/model/

2. Insert specific function into model file.

E.g. ./beatlex/Beatlex.py

``` python
class Beatlex(DMmodel):
    def __init__(self):
        pass
    
    # design for create model object
    def __create__(cls):
        pass

    # design for summarization task
    def summarization(self, params=None):
        pass

    # design for DMmodel
    def run(self):
        pass
```

3. Add call function for dynamic import.

E.g. ./beatlex/Beatlex.py

```python
def __call__():
    return Beatlex
```

4. Register model path to `Task`.py file:

E.g. Beatlex realizes code for summarization task.

In ../task/summarization.py

```python
# MODEL_PATH is a global variabel which refers to 'spartan.model
class SumPolicy(Enum):
    '''Registration for path of models who can do summarization task.
    '''
    Beatlex = MODEL_PATH + ".beatlex.Beatlex"
```
