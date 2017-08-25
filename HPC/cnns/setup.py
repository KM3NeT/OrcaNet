#!/usr/bin/env python
from setuptools import setup

setup(
    name='cnns',
    version='1.0',
    description='Runs CNNs',
    url='https://github.com/ViaFerrata/DL_pipeline_TauAppearance/tree/master/HPC/cnns',
    author='Michael Moser',
    author_email='mmoser@km3net.de',
    license='AGPL',
    packages=['models', 'utilities'],
    include_package_data=True,
)

