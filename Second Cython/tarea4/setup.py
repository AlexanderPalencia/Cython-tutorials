from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_2d_convol_kernel.pyx")
)