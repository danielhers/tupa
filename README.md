Transition-based UCCA Parser
============================
TUPA is a transition-based parser for [Universal Conceptual Cognitive Annotation (UCCA)][1].

### Requirements
* Python 3.x
* All [dependencies for DyNet](http://dynet.readthedocs.io/en/latest/python.html)

### Install

Create a Python virtual environment:
    
    virtualenv --python=/usr/bin/python3 venv
    . venv/bin/activate              # on bash
    source venv/bin/activate.csh     # on csh

Install the latest release:

    pip install tupa

Alternatively, install the latest code from GitHub (may be unstable):

    git clone https://github.com/danielhers/tupa
    cd tupa
    python setup.py install

### Train the parser

Having a directory with UCCA passage files
(for example, [the Wiki corpus](https://github.com/huji-nlp/ucca-corpus/tree/master/wiki/pickle)),
run:

    python -m tupa.parse -t <train_dir> -d <dev_dir> -c <model_type> -m <model_filename>

The possible model types are `sparse`, `mlp` and `bilstm`.

### Parse a text file

Run the parser on a text file (here named `example.txt`) using a trained model:

    python -m tupa.parse example.txt -c <model_type> -m <model_filename>

An `xml` file will be created per passage (separate by blank lines in the text file).

### Pre-trained models

To download and extract models pre-trained on the Wiki corpus, run:

    curl --remote-name-all http://www.cs.huji.ac.il/~danielh/ucca/{sparse,mlp,bilstm}-1.2.tar.gz
    tar xvzf sparse-1.2.tar.gz
    tar xvzf mlp-1.2.tar.gz
    tar xvzf bilstm-1.2.tar.gz

Run the parser using any of them:

    python -m tupa.parse example.txt -c sparse -m models/sparse
    python -m tupa.parse example.txt -c mlp -m models/mlp
    python -m tupa.parse example.txt -c bilstm -m models/bilstm
    
### Other languages

To get a French/German model pre-trained on [the *20K Leagues* corpus](https://github.com/huji-nlp/ucca-corpus/tree/master/vmlslm/fr), run:

    curl -O http://www.cs.huji.ac.il/~danielh/ucca/sparse-1.2-{fr,de}.tar.gz
    tar xvzf sparse-1.2-fr.tar.gz
    tar xvzf sparse-1.2-de.tar.gz

Run the parser on a French/German text file, using the French/German spaCy models too:

    export SPACY_MODEL=fr_depvec_web_lg
    python -m tupa.parse exemple.txt -c sparse -m models/sparse-fr

    export SPACY_MODEL=de_core_news_md
    python -m tupa.parse beispiel.txt -c sparse -m models/sparse-de

Author
------
* Daniel Hershcovich: danielh@cs.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](http://www.cs.huji.ac.il/~danielh/acl2017.pdf):

    @InProceedings{hershcovich2017a,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {A Transition-Based Directed Acyclic Graph Parser for UCCA},
      booktitle = {Proc. of ACL},
      year      = {2017},
      pages     = {1127--1138},
      url       = {http://aclweb.org/anthology/P17-1104}
    }

The version of the parser used in the paper is [v1.0](https://github.com/huji-nlp/tupa/releases/tag/v1.0).
To reproduce the experiments from the paper, run in an empty directory (with a new virtualenv):

    pip install "tupa>=1.0,<1.1"
    mkdir pickle models
    curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca_corpus_pickle.tgz | tar xz -C pickle
    curl --remote-name-all http://www.cs.huji.ac.il/~danielh/ucca/{sparse,mlp,bilstm}.tgz
    tar xvzf sparse.tgz
    tar xvzf mlp.tgz
    tar xvzf bilstm.tgz
    python -m spacy download en
    python -m scripts.split_corpus pickle -t 4282 -d 454 -l
    python -m tupa.parse -c sparse -m models/ucca-sparse -Web pickle/test
    python -m tupa.parse -c mlp -m models/ucca-mlp -Web pickle/test
    python -m tupa.parse -c bilstm -m models/ucca-bilstm -Web pickle/test

License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).

[1]: http://github.com/huji-nlp/ucca


[![Build Status (Travis CI)](https://travis-ci.org/danielhers/tupa.svg?branch=master)](https://travis-ci.org/danielhers/tupa)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/danielhers/tupa?svg=true)](https://ci.appveyor.com/project/danielh/tupa)
[![PyPI version](https://badge.fury.io/py/TUPA.svg)](https://badge.fury.io/py/TUPA)
