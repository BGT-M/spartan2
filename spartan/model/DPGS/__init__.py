try:
    from . import c_MDL
except Exception as e:
    import os
    old_dir = os.getcwd()
    dir_ = os.path.abspath(os.path.dirname(__file__))
    os.chdir(dir_)
    import pyximport
    pyximport.install(build_in_temp=False, inplace=True)
    os.chdir(old_dir)
    from . import c_MDL

from .DPGS import DPGSummarizer

def __call__():
    return DPGSummarizer