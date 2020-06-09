try:
    from . import c_MDL
except ImportError as e:
    import os
    old_dir = os.getcwd()
    dir_ = os.path.abspath(os.path.dirname(__file__))
    os.chdir(dir_)
    import pyximport
    pyximport.install()
    os.chdir(old_dir)

from . import summarizer