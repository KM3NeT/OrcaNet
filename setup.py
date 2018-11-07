#!/usr/bin/env python
from setuptools import setup, find_packages

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='cnns',
    version='1.0',
    description='Runs CNNs',
    url='https://git.km3net.de/mmoser/OrcaNet',
    author='Michael Moser',
    author_email='mmoser@km3net.de',
    license='AGPL',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
)
