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
    pip install .

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

To download and extract [a model pre-trained on the Wiki corpus](https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10.tar.gz), run:

    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10.tar.gz
    tar xvzf ucca-bilstm-1.3.10.tar.gz

Run the parser using the model:

    python -m tupa example.txt -m models/ucca-bilstm
    
### Other languages

To get [a model](https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10-fr.tar.gz) pre-trained on the [French *20K Leagues* corpus](https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_French-20K)
or [a model](https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10-de.tar.gz) pre-trained on the [German *20K Leagues* corpus](https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_German-20K), run:

    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10-fr.tar.gz
    tar xvzf ucca-bilstm-1.3.10-fr.tar.gz
    curl -LO https://github.com/huji-nlp/tupa/releases/download/v1.3.10/ucca-bilstm-1.3.10-de.tar.gz
    tar xvzf ucca-bilstm-1.3.10-de.tar.gz

Run the parser on a French/German text file (separate passages by blank lines):

    python -m tupa exemple.txt -m models/ucca-bilstm-fr --lang fr
    python -m tupa beispiel.txt -m models/ucca-bilstm-de --lang de

### Using BERT embeddings

It's possible to use BERT embeddings instead of the standard not-context-aware embeddings.
To use them pass the `--use-bert` argument in the relevant command and install the packages in  requirements.bert.txt:

    python -m pip install -r requirements.bert.txt

See the possible config options in `config.py` (relevant configs are with the prefix `bert`).

### Using BERT embeddings: Multilingual training
It's possible, when using the BERT embeddings, to train a multilingual model which can leverage
cross-lingual transfer and improve results on low-resource languages.
To train in the multilingual settings you need to:
1) Use BERT embeddings by passing the `--use-bert` argument.
2) Use the BERT multilingual model by passing the argument`--bert-model=bert-base-multilingual-cased`
3) Pass the `--bert-multilingual=0` argument.
4) Make sure the UCCA passages files have the `lang` property. See the script 'set_lang' in the package `semstr`.


Author
------
* Daniel Hershcovich: daniel.hershcovich@gmail.com

Contributors
------------
* Ofir Arviv: ofir.arviv@mail.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](http://aclweb.org/anthology/P17-1104):

    @InProceedings{hershcovich2017a,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {A Transition-Based Directed Acyclic Graph Parser for {UCCA}},
      booktitle = {Proc. of ACL},
      year      = {2017},
      pages     = {1127--1138},
      url       = {http://aclweb.org/anthology/P17-1104}
    }

The version of the parser used in the paper is [v1.0](https://github.com/huji-nlp/tupa/releases/tag/v1.0).
To reproduce the experiments, run:

    curl -L https://raw.githubusercontent.com/huji-nlp/tupa/master/experiments/acl2017.sh | bash
    

If you use the French, German or multitask models, please cite
[the following paper](http://aclweb.org/anthology/P18-1035):

    @InProceedings{hershcovich2018multitask,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {Multitask Parsing Across Semantic Representations},
      booktitle = {Proc. of ACL},
      year      = {2018},
      pages     = {373--385},
      url       = {http://aclweb.org/anthology/P18-1035}
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
[![Build Status (Docs)](https://readthedocs.org/projects/tupa/badge/?version=latest)](http://tupa.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/TUPA.svg)](https://badge.fury.io/py/TUPA)
