from setuptools import setup, Extension
import numpy as np

# Define extension module
pykdtree = Extension(
    'src.utils.libkdtree.pykdtree.kdtree',  # Extension name
    sources=[
        'src/utils/libkdtree/pykdtree/kdtree.c',     # C source files
        'src/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',  # Language of the extension
    extra_compile_args={'gcc': ['-std=c99', '-O3', '-fopenmp']},  # Compiler arguments
    extra_link_args=['-lgomp'],  # Linker arguments
    include_dirs=[np.get_include()]  # Include directories, including numpy
)

# Setup configuration
setup(
    name='pykdtree',  # Package name
    version='1.0',    # Package version
    ext_modules=[pykdtree],  # List of extension modules to build and install
    install_requires=['numpy'],  # Dependencies required for installation
)
