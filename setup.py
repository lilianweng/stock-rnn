#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='stock',
    version='0.1',
    description='Aloha!',
    url='http://github.com/lilianweng/stock-rnn',
    author='Lilian Weng',
    author_email='lilian.wengweng@gmail.com',
    packages=find_packages(exclude=['checkpoints', 'logs', 'images']),
)
