#!/usr/bin/env python

from distutils.core import setup
import os

setup(name='TUPA',
      version='1.0',
      description='Transition-based UCCA Parser',
      author='Daniel Hershcovich',
      author_email='danielh@cs.huji.ac.il',
      url='http://www.cs.huji.ac.il/~oabend/ucca.html',
      packages=['ucca', 'scheme', 'src', 'smatch', 'tupa', 'classifiers', 'nn', 'linear', 'features', 'states'],
      package_dir={
          'ucca': os.path.join('ucca', 'ucca'),
          'scheme': 'scheme',
          'src': os.path.join('scheme', 'amr', 'src'),
          'smatch': os.path.join('scheme', 'smatch'),
          'tupa': 'tupa',
          'classifiers': os.path.join('tupa', 'classifiers'),
          'linear': os.path.join('tupa', 'classifiers', 'linear'),
          'nn': os.path.join('tupa', 'classifiers', 'nn'),
          'features': os.path.join('tupa', 'features'),
          'states': os.path.join('tupa', 'states'),
          },
      package_data={'src': ['amr.peg']},
      )
