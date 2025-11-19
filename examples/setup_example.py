"""
Example setup.py for building a C extension with SWIG

To build:
    python setup.py build_ext --inplace

This will create:
    - _hadamardKernel.so (on Mac/Linux) 
    - _hadamardKernel.pyd (on Windows)
"""

from setuptools import setup, Extension
import numpy as np

# Define the C extension module
hadamard_module = Extension(
    '_hadamardKernel',  # Name of the compiled module (Python will import this)
    sources=[
        'hadamard_wrap.c',      # SWIG-generated wrapper (create with: swig -python hadamard.i)
        'hadamard_kernel.c',    # Your C implementation
    ],
    include_dirs=[np.get_include()],  # Include NumPy headers
    extra_compile_args=['-O3', '-march=native'],  # Optimization flags
)

setup(
    name='hadamardKernel',
    version='1.0',
    description='Fast Walsh-Hadamard Transform C Extension',
    ext_modules=[hadamard_module],
    py_modules=['hadamardKernel'],  # The Python wrapper created by SWIG
)
