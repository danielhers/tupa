Transition-based Meaning Representation Parser
==============================================
TUPA is a transition-based parser used as a baseline system in the
[CoNLL 2019 Shared Task on Cross-Framework Meaning Representation Parsing](http://mrp.nlpl.eu/).
It was originally built for Universal Conceptual Cognitive Annotation (UCCA),
and extended to support DM, PSD, EDS and AMR.

### Requirements
* Python 3.6+

### Install

Install the latest code from GitHub:

    pip install git+https://github.com/danielhers/tupa.git@mrp

### Train the parser

Having a directory with MRP graph files
(for example, [the MRP UCCA data](http://svn.nlpl.eu/mrp/2019/public/ucca.tgz)),
run:

    python -m tupa -t <train_dir> -d <dev_dir> -m <model_filename>

Alternatively, download any of the pre-trained models from https://github.com/danielhers/tupa/releases/tag/mrp2019.

### Parse a text file

Preprocess a text file (here named `example.txt`) using [UDPipe](http://ufal.mff.cuni.cz/udpipe):

    udpipe --tag --parse --input horizontal --tokenizer "ranges;presegmented;normalized_spaces" --output conllu english-ewt-ud-2.4-190531.udpipe < example.txt > example.conllu

Convert the output to `mrp` using [mtool](https://github.com/cfmrp/mtool):
    
    tool/main.py --read conllu --write mrp < example.conllu > example.mrp

Run the parser using a trained model:

    python -m tupa example.mrp -m <model_filename>

An `mrp` file will be created per instance (separate by newlines in the text file).

If you already have a preprocessed `mrp` file (for example, from the shared task data), then there is no need to run UDPipe and mtool.

Author
------
* Daniel Hershcovich: daniel.hershcovich@gmail.com

Contributors
------------
* Ofir Arviv: ofir.arviv@mail.huji.ac.il


Citation
--------
If you make use of this software, please cite [the following paper](https://www.aclweb.org/anthology/K19-2002):

    @InProceedings{hershcovich-arviv-2019-tupa,
      author    = {Hershcovich, Daniel  and  Arviv, Ofir},
      title     = {{TUPA} at {MRP} 2019: A Multi-Task Baseline Syste},
      booktitle = {Proc. of CoNLL MRP Shared Task},
      year      = {2019},
      pages     = {28--39},
      url       = {https://www.aclweb.org/anthology/K19-2002},
      doi       = {10.18653/v1/K19-2002}
    }


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).


[![Build Status (Travis CI)](https://travis-ci.com/danielhers/tupa.svg?branch=mrp)](https://travis-ci.com/danielhers/tupa)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/danielhers/tupa/branch/mrp?svg=true)](https://ci.appveyor.com/project/danielh/tupa/branch/mrp)
