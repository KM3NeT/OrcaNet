#!/usr/bin/env python
from setuptools import setup

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='orcanet',
    description='Runs Neural Networks for usage in the KM3NeT project',
    url='https://git.km3net.de/ml/OrcaNet',
    author='Michael Moser, Stefan Reck',
    author_email='mmoser@km3net.de, michael.m.moser@fau.de, stefan.reck@fau.de',
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
            'summarize=orcanet.utilities.summarize_training:main',
            'orcatrain=orcanet_contrib.parser_orcatrain:main',
            'orcapred=orcanet_contrib.parser_orcapred:main',
        ]
    },

)

__author__ = 'Michael Moser and Stefan Reck'
