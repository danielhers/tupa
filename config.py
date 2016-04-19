import argparse

import numpy as np

from ucca import convert


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

CLASSIFIERS = ("sparse", "dense", "nn")


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        argparser = argparse.ArgumentParser(description="""Transition-based parser for UCCA.""")
        argparser.add_argument("passages", nargs="*", help="passage files/directories to test on/parse")
        argparser.add_argument("-t", "--train", nargs="+", default=(), help="passage files/directories to train on")
        argparser.add_argument("-d", "--dev", nargs="+", default=(), help="passage files/directories to tune on")
        argparser.add_argument("-F", "--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        argparser.add_argument("-m", "--model", help="model file to load/save")
        argparser.add_argument("-w", "--wordvectors", default=100, help="dimensions for random init, or file to load")
        argparser.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=CLASSIFIERS[0], help="model type")
        argparser.add_argument("-o", "--outdir", default=".", help="output directory for parsed files")
        argparser.add_argument("-f", "--format", choices=convert.CONVERTERS, help="output format for parsed files")
        argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
        argparser.add_argument("-O", "--log", default="parse.log", help="output log file")
        argparser.add_argument("-I", "--iterations", type=int, default=1, help="number of training iterations")
        argparser.add_argument("-b", "--binary", action="store_true", help="read and write passages in Pickle")
        argparser.add_argument("-W", "--nowrite", action="store_true", help="do not write parsed passages to file")
        argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate parsed passages")
        argparser.add_argument("-v", "--verbose", action="store_true", help="detailed parse output")
        argparser.add_argument("-s", "--sentences", action="store_true", help="separate passages to sentences")
        argparser.add_argument("-a", "--paragraphs", action="store_true", help="separate passages to paragraphs")
        argparser.add_argument("-r", "--learningrate", type=float, default=1.0, help="rate for model weight updates")
        argparser.add_argument("-D", "--decayfactor", type=float, default=1.0, help="learning rate decay per iteration")
        argparser.add_argument("-i", "--importance", type=float, default=2.0, help="learning rate factor for Swap")
        argparser.add_argument("-u", "--minupdate", type=int, default=5, help="minimum #updates for using a feature")
        argparser.add_argument("-L", "--nolinkage", action="store_true", help="ignore linkage nodes and edges")
        argparser.add_argument("-M", "--noimplicit", action="store_true", help="ignore implicit nodes and edges")
        argparser.add_argument("-R", "--noremote", action="store_true", help="ignore remote edges")
        argparser.add_argument("-S", "--noswap", action="store_true", help="disable Swap transitions entirely")
        argparser.add_argument("-C", "--constraints", action="store_true", help="constrained inference by UCCA rules")
        argparser.add_argument("--devscores", help="output CSV file for dev scores")
        argparser.add_argument("--testscores", help="output CSV file for test scores")
        argparser.add_argument("--checkloops", action="store_true", help="abort if the parser enters a state loop")
        argparser.add_argument("--verify", action="store_true", help="verify oracle reproduces original passage")
        argparser.add_argument("--compoundswap", action="store_true", help="enable compound swap")
        argparser.add_argument("--maxnodes", type=float, default=3.0, help="maximum non-terminal/terminal ratio")
        argparser.add_argument("--maxheight", type=int, default=20, help="maximum graph height")
        argparser.add_argument("--seed", type=int, default=1, help="random number generator seed")
        self.args = argparser.parse_args(args if args else None)

        assert self.args.passages or self.args.train,\
            "Either passages or --train is required (use -h for help)"
        assert self.args.model or self.args.train or self.args.folds,\
            "Either --model or --train or --folds is required"
        assert not (self.args.train or self.args.dev) or self.args.folds is None,\
            "--train and --dev are incompatible with --folds"
        assert self.args.train or not self.args.dev,\
            "--dev is only possible together with --train"
        assert not (self.args.binary and self.args.format),\
            "--binary and --format are incompatible"

        self.verbose = self.args.verbose
        self.line_end = "\n" if self.verbose else " "  # show all in one line unless verbose
        self._log_file = None

        self.sentences = self.args.sentences
        self.paragraphs = self.args.paragraphs
        assert not (self.sentences and self.paragraphs),\
            "--sentences and --paragraphs are incompatible"
        self.split = self.sentences or self.paragraphs

        self.iterations = self.args.iterations
        self.dev_scores = self.args.devscores
        self.test_scores = self.args.testscores
        self.learning_rate = self.args.learningrate
        self.decay_factor = self.args.decayfactor
        self.importance = self.args.importance
        self.min_update = self.args.minupdate
        self.check_loops = self.args.checkloops
        self.verify = self.args.verify
        self.compound_swap = self.args.compoundswap
        self.no_swap = self.args.noswap
        assert not (self.compound_swap and self.no_swap),\
            "--compoundswap and --noswap are incompatible"
        self.max_nodes_ratio = self.args.maxnodes
        self.max_height = self.args.maxheight
        self.no_linkage = self.args.nolinkage
        self.no_implicit = self.args.noimplicit
        self.no_remote = self.args.noremote
        self.constraints = self.args.constraints
        self.word_vectors = self.args.wordvectors
        np.random.seed(self.args.seed)
        self.random = np.random

    def log(self, message):
        if self._log_file is None:
            self._log_file = open(self.args.log, "w")
        print(message, file=self._log_file, flush=True)
        if self.verbose:
            print(message)

    def close(self):
        if self._log_file is not None:
            self._log_file.close()

    def __str__(self):
        return " ".join("%s=%s" % item for item in sorted(vars(self.args).items()))
