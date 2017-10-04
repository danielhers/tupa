#!/bin/bash
set -xe

pip install pypandoc twine
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
