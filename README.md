Transition-based Meaning Representation Parser
==============================================
TUPA is a transition-based parser used as a baseline system in the
[CoNLL 2019 on Cross-Framework Meaning Representation Parsing](http://mrp.nlpl.eu/).
It was originally built for Universal Conceptual Cognitive Annotation (UCCA),
and extended to support DM, PSD, EDS and AMR.

### Requirements
* Python 3.6+

### Install

Create a Python virtual environment. For example, on Linux:
    
    virtualenv --python=/usr/bin/python3 venv
    . venv/bin/activate              # on bash
    source venv/bin/activate.csh     # on csh

Install the latest code from GitHub (may be unstable):

    git clone https://github.com/danielhers/tupa --branch=mrp
    cd tupa
    pip install .

### Train the parser

Having a directory with MRP graph files
(for example, [the MRP sample](http://svn.nlpl.eu/mrp/2019/public/sample.tgz)),
run:

    python -m tupa -t <train_dir> -d <dev_dir> -m <model_filename>

### Parse a text file

Run the parser on a text file (here named `example.txt`) using a trained model:

    python -m tupa example.txt -m <model_filename>

An `mrp` file will be created per instance (separate by blank lines in the text file).

Author
------
* Daniel Hershcovich: daniel.hershcovich@gmail.com

Contributors
------------
* Ofir Arviv: ofir.arviv@mail.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](http://aclweb.org/anthology/P18-1035):

    @InProceedings{hershcovich2018multitask,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {Multitask Parsing Across Semantic Representations},
      booktitle = {Proc. of ACL},
      year      = {2018},
      pages     = {373--385},
      url       = {http://aclweb.org/anthology/P18-1035}
    }


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).


[![Build Status (Travis CI)](https://travis-ci.org/danielhers/tupa.svg?branch=mrp)](https://travis-ci.org/danielhers/tupa)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/danielhers/tupa/branch/mrp?svg=true)](https://ci.appveyor.com/project/danielh/tupa/branch/mrp)
