#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup
from os import path

# short/long description
short_desc ='A lightweight genetic algorithm module for optimization problems'
here = path.abspath(path.dirname(__file__))
try:
    with open(path.join(here, 'README.md'),'r',encoding='utf-8') as f:
        long_desc = '\n' + f.read()
except FileNotFoundError:
    long_desc = short_desc

setup(
    name='meltingpot',
    version='1.0.0',
    description=short_desc,
    author='andrea capitanelli',
    author_email='andrea.capitanelli@gmail.com',
    maintainer='andrea capitanelli',
    maintainer_email='andrea.capitanelli@gmail.com',
    url='https://github.com/acapitanelli/meltingpot',
    install_requires=['numpy'],
    packages=['meltingpot'],
    long_description=long_desc,
    long_description_content_type='text/markdown',
    keywords='optimization genetic algorithm constraints',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
