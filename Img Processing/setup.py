from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("iterate_over_img.pyx"))
