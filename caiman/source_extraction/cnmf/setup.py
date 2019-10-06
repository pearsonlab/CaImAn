# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("timebuffer", ["timebuffer.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=['cblas', 'blas', 'lapack']
              )
]
setup(
    ext_modules=cythonize(extensions),

)