#!/usr/bin/env python
from setuptools import setup

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

with open("Readme.rst") as fh:
    long_description = fh.read()

setup(
    name='orcanet',
    description='Runs Neural Networks for usage in the KM3NeT project',
    long_description=long_description,
    url='https://git.km3net.de/ml/OrcaNet',
    author='Michael Moser, Stefan Reck',
    author_email='stefan.reck@fau.de, mmoser@km3net.de, michael.m.moser@fau.de',
    license='AGPL',
    install_requires=requirements,
    packages=[
        "orcanet", "orcanet_contrib"
    ],
    include_package_data=True,

    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': 'orcanet/version.txt',
        'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],

    entry_points={
        'console_scripts': [
            'orcanet=orcanet.parser:main',
            'orcatrain=orcanet_contrib.parser_orcatrain:main',  # TODO deprectated
            'orcapred=orcanet_contrib.parser_orcapred:main',  # TODO deprectated
        ]
    },

)

__author__ = 'Michael Moser and Stefan Reck'
