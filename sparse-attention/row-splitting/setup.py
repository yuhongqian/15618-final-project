import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

ext_modules = []

if torch.cuda.is_available():
    ext = CUDAExtension(
        "row_split_spmm",
        [
            "natives/row_split_native.cpp",
            "pytorch_block_sparse/native/row_split.cu",
        ],
        extra_compile_args=["-I", rootdir],
    )
    ext_modules = [ext]
else:
    print("WARNING: torch cuda seems unavailable, emulated features only will be available.")

setup(
    name="row_split_spmm",
    description="PyTorch extension for fast sparse matrices computation,"
                " drop in replacement for torch.nn.Linear.",
    long_description="row_split_spmm is a PyTorch extension for fast block sparse matrices computation,"
                     " drop in replacement for torch.nn.Linear",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="PyTorch,sparse,matrices,machine learning",
    author="HongChien Yu",
    author_email="hongqiay@cs.cmu.edu",
    license='BSD 3-Clause "New" or "Revised" License',
    packages=["row_split_spmm"],
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
