#!/usr/bin/env python3

from setuptools import setup, Extension
from distutils.sysconfig import get_python_version

kay_inc = ["/home/kirkm/kay/include"]
iv_lib = []
build = "/home/kirkm/projects/smlp/smd/utils/poly/build/src"

boost = "boost_python" + get_python_version().replace(".", "")

libs = [
    "boost",
    "smlp",
    "iv",
    "kjson",
    "hdf5_cpp",
    "hdf5",
    "z3",
    "flint",
    "mpfr",
    "gmp",
]

setup(
    name="smlp",
    version="0.10.1",
    author="Franz Brauße",
    author_email="franz.brausse@manchester.ac.uk",
    url="https://github.com/fbrausse/smlp",
    license="Apache-2.0",
    packages=["smlp"],
    package_dir={"": "python"},
    ext_modules=[
        Extension(
            "smlp.libsmlp",
            ["src/libsmlp.cc"],
            include_dirs=["include", build] + kay_inc,
            libraries=libs,
            library_dirs=[build] + iv_lib,
            runtime_library_dirs=iv_lib,
            extra_compile_args=["-std=c++20", "-O2"],
            undef_macros=["NDEBUG"],
            language="c++",
        )
    ],
)
