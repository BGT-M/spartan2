from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("c_MDL",
        ["c_MDL.pyx"],
    )
]

setup(
    name="c_MDL",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)
