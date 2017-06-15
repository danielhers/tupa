Transition-based UCCA Parser [![Build Status](https://travis-ci.org/danielhers/tupa.svg?branch=master)](https://travis-ci.org/danielhers/tupa)
============================
TUPA is a transition-based parser for [Universal Conceptual Cognitive Annotation (UCCA)][1].

### Requirements
* Python 3.x
* All [dependencies for DyNet](http://dynet.readthedocs.io/en/latest/python.html)

### Build

Install the required modules:
    
    virtualenv --python=/usr/bin/python3 venv
    . venv/bin/activate              # on bash
    source venv/bin/activate.csh     # on csh
    ./install_dependencies.sh
    python setup.py install

### Train the parser

Having a directory with UCCA passage files
(for example, [the Wiki corpus](https://github.com/huji-nlp/ucca-corpus/tree/master/wiki/pickle)),
run:

    python tupa/parse.py -t <train_dir> -d <dev_dir> -c <model_type> -m <model_filename>

The possible model types are `sparse`, `mlp` and `bilstm`.

### Parse a text file

Run the parser on a text file (here named `example.txt`) using a trained model:

    python tupa/parse.py example.txt -c <model_type> -m <model_filename>

An `xml` file will be created per passage (separate by blank lines in the text file).

### Pre-trained models

To download and extract the pre-trained models, run:

    wget http://www.cs.huji.ac.il/~danielh/ucca/{sparse,mlp,bilstm}.tar.gz
    tar xvzf sparse.tar.gz
    tar xvzf mlp.tar.gz
    tar xvzf bilstm.tar.gz

Run the parser using any of them:

    python tupa/parse.py example.txt -c sparse -m models/sparse
    python tupa/parse.py example.txt -c mlp -m models/mlp
    python tupa/parse.py example.txt -c bilstm -m models/bilstm

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
