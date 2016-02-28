import argparse

from ucca import convert


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        argparser = argparse.ArgumentParser(description="""Transition-based parser for UCCA.""")
        argparser.add_argument("passages", nargs="*", default=(),
                               help="passage files/directories to test on/parse")
        argparser.add_argument("-t", "--train", nargs="+", default=(),
                               help="passage files/directories to train on")
        argparser.add_argument("-d", "--dev", nargs="+", default=(),
                               help="passage files/directories to tune on")
        argparser.add_argument("-F", "--folds", type=int, choices=range(3, 11),
                               help="number of folds for k-fold cross validation")
        argparser.add_argument("-m", "--model",
                               help="model file to load/save")
        argparser.add_argument("-o", "--outdir", default=".",
                               help="output directory for parsed files")
        argparser.add_argument("-f", "--format", choices=convert.CONVERTERS,
                               help="output format for parsed files, if not UCCA format")
        argparser.add_argument("-p", "--prefix", default="",
                               help="output filename prefix")
        argparser.add_argument("-L", "--log", default="parse.log",
                               help="output log file")
        argparser.add_argument("-O", "--devscores", default="dev_scores.csv",
                               help="output file for dev scores")
        argparser.add_argument("-I", "--iterations", type=int, default=1,
                               help="number of training iterations")
        argparser.add_argument("-b", "--binary", action="store_true",
                               help="read and write passages in Pickle binary format, not XML")
        argparser.add_argument("-e", "--evaluate", action="store_true",
                               help="show evaluation results on parsed passages")
        argparser.add_argument("-v", "--verbose", action="store_true",
                               help="display detailed information while parsing")
        argparser.add_argument("-s", "--sentences", action="store_true",
                               help="separate passages to sentences and parse each one separately")
        argparser.add_argument("-a", "--paragraphs", action="store_true",
                               help="separate passages to paragraphs and parse each one separately")
        argparser.add_argument("-r", "--learningrate", type=float, default=1.0,
                               help="learning rate for the model weight updates")
        argparser.add_argument("-D", "--decayfactor", type=float, default=0.9,
                               help="learning rate decay factor at each training iteration")
        argparser.add_argument("-i", "--importance", type=float, default=2.0,
                               help="learning rate factor at swap transitions")
        argparser.add_argument("-u", "--minupdate", type=int, default=5,
                               help="minimum updates a feature must have before being used")
        argparser.add_argument("-l", "--checkloops", action="store_true",
                               help="abort if the parser reaches the exact same state as it did before")
        argparser.add_argument("-V", "--verify", action="store_true",
                               help="verify oracle successfully reproduces the passage")
        argparser.add_argument("-c", "--compoundswap", action="store_true",
                               help="enable compound swap")
        argparser.add_argument("-N", "--maxnodes", type=float, default=3.0,
                               help="maximum ratio between non-terminal to terminal nodes")
        argparser.add_argument("-M", "--multiedge", action="store_true",
                               help="allow multiple edges between the same nodes (with different tags)")
        argparser.add_argument("-n", "--nolinkage", action="store_true",
                               help="ignore linkage nodes and edges during both train and test")
        argparser.add_argument("-S", "--noswap", action="store_true",
                               help="disable swap transitions entirely")
        argparser.add_argument("-C", "--constraints", action="store_true",
                               help="use constrained inference according to UCCA rules")
        self.args = argparser.parse_args(args if args else None)

        assert self.args.passages or self.args.train,\
            "Either passages or --train is required"
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
        self.multiple_edges = self.args.multiedge
        self.no_linkage = self.args.nolinkage
        self.constraints = self.args.constraints

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
        return " ".join("%s=%s" % item for item in vars(self.args).items())
