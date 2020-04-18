from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("blokus",  ["blokus/envs/blokus_env.py"]),
    # Extension("mymodule2",  ["mymodule2.py"]),
    #   ... all your modules that need be compiled ...
]
setup(
    name='Blokus',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
