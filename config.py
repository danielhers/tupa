import argparse

import numpy as np

from ucca import convert


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

# Multiple choice options: the first one is always the default
CLASSIFIERS = ("sparse", "dense", "nn")
OPTIMIZERS = ("adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax")
OBJECTIVES = ("categorical_crossentropy", "hinge", "squared_hinge")


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        argparser = argparse.ArgumentParser(description="""Transition-based parser for UCCA.""")
        argparser.add_argument("passages", nargs="*", help="passage files/directories to test on/parse")
        argparser.add_argument("-m", "--model", help="model file to load/save")
        argparser.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=CLASSIFIERS[0], help="model type")
        argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate parsed passages")
        argparser.add_argument("-v", "--verbose", action="store_true", help="detailed parse output")
        group = argparser.add_mutually_exclusive_group()
        group.add_argument("-s", "--sentences", action="store_true", help="separate passages to sentences")
        group.add_argument("-a", "--paragraphs", action="store_true", help="separate passages to paragraphs")
        group = argparser.add_argument_group(title="Training parameters")
        group.add_argument("-t", "--train", nargs="+", default=(), help="passage files/directories to train on")
        group.add_argument("-d", "--dev", nargs="+", default=(), help="passage files/directories to tune on")
        group.add_argument("-I", "--iterations", type=int, default=1, help="number of training iterations")
        group.add_argument("--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        group.add_argument("--seed", type=int, default=1, help="random number generator seed")
        group = argparser.add_argument_group(title="Output files")
        group.add_argument("-o", "--outdir", default=".", help="output directory for parsed files")
        group.add_argument("-p", "--prefix", default="", help="output filename prefix")
        group.add_argument("-W", "--nowrite", action="store_true", help="do not write parsed passages to file")
        group.add_argument("-O", "--log", default="parse.log", help="output log file")
        group.add_argument("--devscores", help="output CSV file for dev scores")
        group.add_argument("--testscores", help="output CSV file for test scores")
        group = argparser.add_argument_group(title="Structural constraints")
        group.add_argument("-L", "--nolinkage", action="store_true", help="ignore linkage nodes and edges")
        group.add_argument("-M", "--noimplicit", action="store_true", help="ignore implicit nodes and edges")
        group.add_argument("-R", "--noremote", action="store_true", help="ignore remote edges")
        group.add_argument("-C", "--constraints", action="store_true", help="constrained inference by UCCA rules")
        group.add_argument("--maxnodes", type=float, default=3.0, help="maximum non-terminal/terminal ratio")
        group.add_argument("--maxheight", type=int, default=20, help="maximum graph height")
        group = argparser.add_mutually_exclusive_group()
        group.add_argument("-S", "--noswap", action="store_true", help="disable Swap transitions entirely")
        group.add_argument("--compoundswap", action="store_true", help="enable compound swap")
        group = argparser.add_argument_group(title="Sanity checks")
        group.add_argument("--checkloops", action="store_true", help="abort if the parser enters a state loop")
        group.add_argument("--verify", action="store_true", help="verify oracle reproduces original passage")
        group = group.add_mutually_exclusive_group()
        group.add_argument("-b", "--binary", action="store_true", help="read and write passages in Pickle")
        group.add_argument("-f", "--format", choices=convert.CONVERTERS, help="output format for parsed files")
        group = argparser.add_argument_group(title="Perceptron parameters")
        group.add_argument("--learningrate", type=float, default=1.0, help="rate for model weight updates")
        group.add_argument("--decayfactor", type=float, default=1.0, help="learning rate decay per iteration")
        group.add_argument("--importance", type=float, default=2.0, help="learning rate factor for Swap")
        group.add_argument("--minupdate", type=int, default=5, help="minimum #updates for using a feature")
        group = argparser.add_argument_group(title="Neural network parameters")
        group.add_argument("-w", "--wordvectors", default=100, help="dimensions for random init, or file to load")
        group.add_argument("--tagdim", type=int, default=10, help="dimension for POS tag embeddings")
        group.add_argument("--labeldim", type=int, default=10, help="dimension for edge label embeddings")
        group.add_argument("--punctdim", type=int, default=2, help="dimension for separator punctuation embeddings")
        group.add_argument("--gapdim", type=int, default=2, help="dimension for gap type embeddings")
        group.add_argument("--layerdim", type=int, default=100, help="dimension for hideen layer")
        group.add_argument("--maxlabels", type=int, default=100, help="maximum number of actions to allow")
        group.add_argument("--batchsize", type=int, help="if given, fit model every this many updates")
        group.add_argument("--minibatchsize", type=int, default=200, help="mini-batch size for optimization")
        group.add_argument("--nbepochs", type=int, default=5, help="number of epochs for optimization")
        group.add_argument("--optimizer", choices=OPTIMIZERS, default=OPTIMIZERS[0], help="algorithm for optimization")
        group.add_argument("--loss", choices=OBJECTIVES, default=OBJECTIVES[0], help="loss function for optimization")
        self.args = argparser.parse_args(args if args else None)

        assert self.args.passages or self.args.train,\
            "Either passages or --train is required (use -h for help)"
        assert self.args.model or self.args.train or self.args.folds,\
            "Either --model or --train or --folds is required"
        assert not (self.args.train or self.args.dev) or self.args.folds is None,\
            "--train and --dev are incompatible with --folds"
        assert self.args.train or not self.args.dev,\
            "--dev is only possible together with --train"

        self.verbose = self.args.verbose
        self.line_end = "\n" if self.verbose else " "  # show all in one line unless verbose
        self._log_file = None
        self.sentences = self.args.sentences
        self.paragraphs = self.args.paragraphs
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
        self.max_nodes_ratio = self.args.maxnodes
        self.max_height = self.args.maxheight
        self.no_linkage = self.args.nolinkage
        self.no_implicit = self.args.noimplicit
        self.no_remote = self.args.noremote
        self.constraints = self.args.constraints
        self.word_vectors = self.args.wordvectors
        self.tag_dim = self.args.tagdim
        self.label_dim = self.args.labeldim
        self.punct_dim = self.args.punctdim
        self.gap_dim = self.args.gapdim
        self.layer_dim = self.args.layerdim
        self.max_num_labels = self.args.maxlabels
        self.batch_size = self.args.batchsize
        self.minibatch_size = self.args.minibatchsize
        self.nb_epochs = self.args.nbepochs
        self.optimizer = self.args.optimizer
        self.loss = self.args.loss
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
