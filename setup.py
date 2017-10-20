#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='drag_utils',
    version='0.1',
    description='Fluid-particle drag utilities',
    packages=find_packages(),
    install_requires=['numpy']
)
