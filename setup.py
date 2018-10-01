#!/usr/bin/env python

import os
import sys
import glob
import tensorflow as tf
from setuptools import setup

sys.path.append("shoop")

from cpp_extension import (  # NOQA
    BuildExtension, CppExtension, CUDAExtension
)


def construct_op(name):
    path = os.path.join("shoop", name)
    cpp_files = glob.glob(os.path.join(path, "*.cc"))
    cuda_files = glob.glob(os.path.join(path, "*.cu"))

    flags = []
    if sys.platform == "darwin":
        flags += ["-mmacosx-version-min=10.9"]

    if tf.test.is_built_with_cuda() and len(cuda_files):
        flags.append("-DGOOGLE_CUDA=1")
        return CUDAExtension(
            "exoplanet." + name + "_op",
            cpp_files + cuda_files,
            include_dirs=[path, "include"],
            extra_compile_args={
                "cxx": flags,
                "nvcc": flags + ["--expt-relaxed-constexpr"],
            },
            extra_link_args=flags,
        )

    return CppExtension(
        "shoop." + name,
        cpp_files,
        include_dirs=[path, "include"],
        extra_compile_args=flags,
        extra_link_args=flags,
    )


extensions = [
    construct_op("ops"),
]

setup(
    name="shoop",
    license="MIT",
    packages=["shoop"],
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=True,
)
