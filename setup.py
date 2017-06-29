#!/usr/bin/env python3
# coding: utf-8

import io
from setuptools import setup, find_packages

# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="tf_G",
    version="0.1",
    description="Python's Tensorflow Graph Library",
    author="garciparedes",
    author_email="sergio@garciparedes.me",
    url="http://tf_G.readthedocs.io/en/latest/",
    download_url="https://github.com/garciparedes/tf_G",
    keywords=[
        "tfg", "bigdata", "tensorflow",
        "graph theory", "pagerank", "university of valladolid",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.13",
        "pandas>=0.20",
        "tensorflow>=1.2.0",
    ],
    tests_require=[
        "pytest"
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    long_description=io.open('README.rst', encoding='utf-8').read(),
    include_package_data=True,
    zip_safe=False,
)
