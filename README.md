Transition-based UCCA Parser [![Build Status](https://travis-ci.org/danielhers/tupa.svg?branch=master)](https://travis-ci.org/danielhers/tupa)
============================
TUPA is a transition-based parser for [Universal Conceptual Cognitive Annotation (UCCA)][1].

### Requirements
* Python 3.x
* [DyNet](https://github.com/clab/dynet)

### Build

Install the required modules:
    
    git submodule update --init --recursive
    virtualenv --python=/usr/bin/python3 .
    . bin/activate  # on bash
    source bin/activate.csh  # on csh
    pip install -r requirements.txt
    python -m spacy.en.download all
    ci/install-dynet.sh
    python ucca/setup.py install
    python setup.py install

### Train the parser

Having a directory with UCCA passage files
(for example, [the Wiki corpus](https://github.com/huji-nlp/ucca-corpus/tree/master/wiki/pickle)),
run:

    python tupa/parse.py -t <train_dir> -d <dev_dir> -m <model_filename>

To specify a model type (`sparse`, `mlp` or `bilstm`),
add `-c <model_type>`.

### Parse a text file

Run the parser on a text file (here named `example.txt`) using a trained model:

    python tupa/parse.py example.txt -m <model_filename>

A file named `example.xml` will be created.

If you specified a model type using `-c` when training the model,
be sure to include it when parsing too.

### Pre-trained models

To download and extract the pre-trained models, run:

    wget http://www.cs.huji.ac.il/~danielh/ucca/{sparse,mlp,bilstm}.tgz
    tar xvzf sparse.tgz
    tar xvzf mlp.tgz
    tar xvzf bilstm.tgz

Run the parser using any of them:

    python tupa/parse.py example.txt -c sparse -m models/ucca-sparse
    python tupa/parse.py example.txt -c mlp -m models/ucca-mlp
    python tupa/parse.py example.txt -c bilstm -m models/ucca-bilstm

Author
------
* Daniel Hershcovich: danielh@cs.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](http://www.cs.huji.ac.il/~danielh/acl2017.pdf):

	@inproceedings{hershcovich2017a,
	  title={A Transition-Based Directed Acyclic Graph Parser for {UCCA}},
	  author={Hershcovich, Daniel and Abend, Omri and Rappoport, Ari},
	  booktitle={Proc. of ACL},
	  year={2017}
	}


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).

[1]: http://github.com/huji-nlp/ucca
