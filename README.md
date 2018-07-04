Transition-based UCCA Parser
============================
TUPA is a transition-based parser for [Universal Conceptual Cognitive Annotation (UCCA)][1].

### Requirements
* Python 3.6

### Install

Create a Python virtual environment. For example, on Linux:
    
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
(for example, [the English Wiki corpus](https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_English-Wiki)),
run:

    python -m tupa -t <train_dir> -d <dev_dir> -c <model_type> -m <model_filename>

The possible model types are `sparse`, `mlp`, and `bilstm`.

### Parse a text file

Run the parser on a text file (here named `example.txt`) using a trained model:

    python -m tupa example.txt -m <model_filename>

An `xml` file will be created per passage (separate by blank lines in the text file).

### Pre-trained models

To download and extract [a model pre-trained on the Wiki corpus](https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3.tar.gz), run:

    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3.tar.gz
    tar xvzf ucca-bilstm-1.3.3.tar.gz

Run the parser using the model:

    python -m tupa example.txt -m models/ucca-bilstm
    
### Other languages

To get [a model](https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3-fr.tar.gz) pre-trained on the [French *20K Leagues* corpus](https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_French-20K)
or [a model](https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3-de.tar.gz) pre-trained on the [German *20K Leagues* corpus](https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_German-20K), run:

    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3-fr.tar.gz
    tar xvzf ucca-bilstm-1.3.3-fr.tar.gz
    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.3/ucca-bilstm-1.3.3-de.tar.gz
    tar xvzf ucca-bilstm-1.3.3-de.tar.gz

Run the parser on a French/German text file (separate passages by blank lines):

    python -m tupa exemple.txt -m models/ucca-bilstm-fr --lang fr
    python -m tupa beispiel.txt -m models/ucca-bilstm-de --lang de

Author
------
* Daniel Hershcovich: danielh@cs.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](http://aclweb.org/anthology/P17-1104):

    @InProceedings{hershcovich2017a,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {A Transition-Based Directed Acyclic Graph Parser for UCCA},
      booktitle = {Proc. of ACL},
      year      = {2017},
      pages     = {1127--1138},
      url       = {http://aclweb.org/anthology/P17-1104}
    }

The version of the parser used in the paper is [v1.0](https://github.com/huji-nlp/tupa/releases/tag/v1.0).
To reproduce the experiments, run:

    curl -L https://raw.githubusercontent.com/huji-nlp/tupa/master/experiments/acl2017.sh | bash
    

If you use the French, German or multitask models, please cite
[the following paper](http://www.cs.huji.ac.il/~danielh/acl2018.pdf):

    @InProceedings{hershcovich2018multitask,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {Multitask Parsing Across Semantic Representations},
      booktitle = {Proc. of ACL},
      year      = {2018},
      url       = {http://www.cs.huji.ac.il/~danielh/acl2018.pdf}
    }

The version of the parser used in the paper is [v1.3.3](https://github.com/huji-nlp/tupa/releases/tag/v1.3.3).
To reproduce the experiments, run:

    curl -L https://raw.githubusercontent.com/huji-nlp/tupa/master/experiments/acl2018.sh | bash


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).

[1]: http://github.com/huji-nlp/ucca


[![Build Status (Travis CI)](https://travis-ci.org/danielhers/tupa.svg?branch=master)](https://travis-ci.org/danielhers/tupa)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/danielhers/tupa?svg=true)](https://ci.appveyor.com/project/danielh/tupa)
[![PyPI version](https://badge.fury.io/py/TUPA.svg)](https://badge.fury.io/py/TUPA)
