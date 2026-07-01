from setuptools import setup, Extension
import os
import sys

# Ensure pybind11 is accessible during execution
try:
    import pybind11
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11

functions_ext = Extension(
    'solver_py',
    sources=[
        'solver_py.cpp',
        'game_theory_solver.cpp',
        'irrigation_solver.cpp',
        '../src/solver_impl.cpp'
    ],
    include_dirs=[
        '../include',
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    language='c++',
    extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']
)

setup(
    name='solver_py',
    version='0.1.0',
    author='Infrastructure Research Core',
    description='High-Performance Computational Module for SARE System Optimization',
    ext_modules=[functions_ext],
    zip_safe=False,
    python_requires=">=3.7",
)