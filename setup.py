#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('USAGE.md', encoding='utf8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf8') as f:
    license = f.read()

setup(
    name='nnrecommend',
    version='0.0.1',
    description='recommender using deep neural networks',
    long_description=readme,
    author='Abel Martínez, Rafael Pérez, Miguel Ibero',
    author_email='abelmart@gmail.com, rafaelapm93@gmail.com, miguel@ibero.me',
    url='http://github.com/miguelibero/aidl-nnrecommend',
    license=license,
    python_requires='>=3.8.0',
    packages=find_packages(exclude=('docs')),
    entry_points={
        'console_scripts': [
            'nnrecommend=nnrecommend.cli:main',
        ],
    },
    install_requires=[
        'pandas~=1.2.4',
        'numpy~=1.20.2',
        'scipy~=1.6.3',
        'torch~=1.8.0',
        'torch-geometric~=1.7.0',
        'torch-sparse~=0.6.9',
        'torch-scatter~=2.0.6',
        'coloredlogs~=15.0',
        'click~=7.1.2',
        'fuzzywuzzy~=0.18.0',
        'python-Levenshtein~=0.12.2'
        'plotly~=4.14.3',
        'scikit-learn~=0.24.2',
        'tensorboard~=2.5.0',
        'scikit-surprise~=1.1.1',
        'ray[default,tune]~=1.3.0'
    ],
    extras_require={
        'test': [
            'pytest~=6.2.3',
        ]
    }
)