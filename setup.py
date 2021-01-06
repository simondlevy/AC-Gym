#!/usr/bin/env python3
'''
Python distutils setup file for ac-gym package.

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
       name='ac_gym',
       version='0.1',
       install_requires=['gym', 'numpy'],
       description='Use NEAT to learn OpenAI Gym environment',
       packages=['ac_gym', 'ac_gym.ptan', 'ac_gym.ptan.common'],
       author='Simon D. Levy',
       author_email='simon.d.levy@gmail.com',
       url='https://github.com/simondlevy/gym-copter',
       license='MIT',
       platforms='Linux; Windows; OS X'
      )
