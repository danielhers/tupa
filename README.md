Transition-based UCCA Parser [![Build Status](https://travis-ci.org/danielhers/tupa.svg?branch=master)](https://travis-ci.org/danielhers/tupa)
============================
TUPA is a transition-based parser for [Universal Conceptual Cognitive Annotation (UCCA)][1].

This Python 3 package provides a parser for UCCA.

Running the parser:
-------------------

Install the required modules and spaCy models:

    virtualenv --python=/usr/bin/python3 .
    . bin/activate  # on bash
    source bin/activate.csh  # on csh
    pip install -r requirements.txt
    python -m spacy.en.download all
    python setup.py install

Download and extract the pre-trained models:

    wget http://www.cs.huji.ac.il/~danielh/ucca/{sparse,dense,mlp,bilstm}.tgz
    tar xvzf sparse.tgz
    tar xvzf dense.tgz
    tar xvzf mlp.tgz
    tar xvzf bilstm.tgz

Run the parser on a text file (here named `example.txt`) using either of the models:

    python tupa/parse.py example.txt -c sparse -m models/ucca-sparse
    python tupa/parse.py example.txt -c dense -m models/ucca-dense
    python tupa/parse.py example.txt -c mlp -m models/ucca-mlp
    python tupa/parse.py example.txt -c bilstm -m models/ucca-bilstm

A file named `example.xml` will be created.

The `tupa` package contains code for a full UCCA parser, currently under construction.

Author
------
* Daniel Hershcovich: danielh@cs.huji.ac.il


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](master/LICENSE.txt)).

[1]: http://github.com/huji-nlp/ucca
