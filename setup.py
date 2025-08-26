from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "hanoi.sim",
        ["hanoi/sim.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"})
)



