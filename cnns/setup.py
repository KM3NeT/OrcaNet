#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='cnns',
    version='1.0',
    description='Runs CNNs',
    url='https://github.com/ViaFerrata/DL_pipeline_TauAppearance/tree/master/HPC/cnns',
    author='Michael Moser',
    author_email='mmoser@km3net.de',
    license='AGPL',
    packages=find_packages(),
    include_package_data=True,
)

