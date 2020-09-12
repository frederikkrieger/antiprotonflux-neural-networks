from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension("integral",
                                    sources=["integral.pyx"],
                                    include_dirs=['./']),
                          annotate=True)
)