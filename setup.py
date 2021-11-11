from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("iterate_img_matrix.pyx")
)