# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import os

import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


install_requires = [
    "numpy>=1.22",
    "attrs>=21.3.0",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "tqdm",
]

tests_require = ["pytest"]

meta = {}
exec(read("ibicus/__meta__.py"), meta)


setuptools.setup(
    # Essential details on the package and its dependencies
    name=meta["__name__"],
    version=meta["__version__"],
    description=meta.get("__description__", ""),
    long_description=read("README.rst"),
    author=meta.get("__author__", "European Centre for Medium-Range Weather Forecasts (ECMWF)"),
    author_email=meta.get("__author_email__", "software.support@ecmwf.int"),
    license="Apache License Version 2.0",
    url="https://github.com/esowc/ibicus",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    zip_safe=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    tests_require=tests_require,
    test_suite="tests",
)
