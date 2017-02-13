import argparse
import sys

import numpy as np

from ucca import convert, constructions


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

    def reload(cls):
        cls.instance = None


class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values is None:
            values = "1"
        try:
            values = int(values)
        except ValueError:
            values = values.count("v") + 1
        setattr(args, self.dest, values)

SPARSE_PERCEPTRON = "sparse"
DENSE_PERCEPTRON = "dense"
MLP_NN = "mlp"
BILSTM_NN = "bilstm"
CLASSIFIERS = (SPARSE_PERCEPTRON, DENSE_PERCEPTRON, MLP_NN, BILSTM_NN)

# Multiple choice options: the first one is always the default
ACTIVATIONS = ("sigmoid", "tanh", "relu", "cube")
INITIALIZATIONS = ("glorot_uniform", "normal", "uniform", "const")
OPTIMIZERS = ("adam", "sgd", "momentum", "adagrad", "adadelta")


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        argparser = argparse.ArgumentParser(description="""Transition-based parser for UCCA.""",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        argparser.add_argument("passages", nargs="*", help="passage files/directories to test on/parse")
        argparser.add_argument("-m", "--model", help="model file to load/save (default: ucca_<model_type>")
        argparser.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=SPARSE_PERCEPTRON, help="model type")
        argparser.add_argument("-B", "--beam", choices=(1,), default=1, help="beam size for beam search (1 for greedy)")
        argparser.add_argument("-e", "--evaluate", action="store_true", help="evaluate parsed passages")
        argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, help="detailed parse output")
        constructions.add_argument(argparser)
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
        group.add_argument("-W", "--no-write", action="store_true", help="do not write parsed passages to files")
        group.add_argument("-l", "--log", help="output log file (default: model filename + .log)")
        group.add_argument("--devscores", help="output CSV file for dev scores (default: model filename + .dev.csv)")
        group.add_argument("--testscores", help="output CSV file for test scores (default: model filename + .test.csv)")
        group = argparser.add_argument_group(title="Structural constraints")
        group.add_argument("--linkage", action="store_true", help="include linkage nodes and edges")
        group.add_argument("--implicit", action="store_true", help="include implicit nodes and edges")
        group.add_argument("--no-remote", action="store_false", dest="remote", help="ignore remote edges")
        group.add_argument("--no-constraints", action="store_false", dest="constraints", help="ignore UCCA rules")
        group.add_argument("--max-nodes", type=float, default=3.0, help="max non-terminal/terminal ratio")
        group.add_argument("--max-height", type=int, default=20, help="max graph height")
        group = argparser.add_mutually_exclusive_group()
        group.add_argument("--no-swap", action="store_false", dest="swap", help="disable Swap transitions entirely")
        group.add_argument("--compound-swap", action="store_true", help="enable compound swap")
        group = argparser.add_argument_group(title="Sanity checks")
        group.add_argument("--check-loops", action="store_true", help="abort if the parser enters a state loop")
        group.add_argument("--verify", action="store_true", help="verify oracle reproduces original passage")
        group = group.add_mutually_exclusive_group()
        group.add_argument("-b", "--binary", action="store_true", help="read and write passages in Pickle")
        group.add_argument("-f", "--format", choices=convert.CONVERTERS, help="output format for parsed files")
        group = argparser.add_argument_group(title="General classifier training parameters")
        group.add_argument("--swap-importance", type=int, default=1, help="learning rate factor for Swap")
        group.add_argument("--early-update", action="store_true", help="move to next example on incorrect prediction")
        group.add_argument("--word-dim-external", type=int, default=300, help="dimension for external word embeddings")
        group.add_argument("--word-vectors", help="file to load external word embeddings from (default: GloVe)")
        group = argparser.add_argument_group(title="Perceptron parameters")
        group.add_argument("--learning-rate", type=float, default=1.0, help="rate for model weight updates")
        group.add_argument("--learning-rate-decay", type=float, default=0.0, help="learning rate decay per iteration")
        group.add_argument("--min-update", type=int, default=5, help="minimum #updates for using a feature")
        group = argparser.add_argument_group(title="Neural network parameters")
        group.add_argument("--update-word-vectors", action="store_true", help="tune the external word embeddings")
        group.add_argument("--word-dim", type=int, default=100, help="dimension for learned word embeddings")
        group.add_argument("--tag-dim", type=int, default=10, help="dimension for POS tag embeddings")
        group.add_argument("--dep-dim", type=int, default=10, help="dimension for dependency relation embeddings")
        group.add_argument("--label-dim", type=int, default=20, help="dimension for edge label embeddings")
        group.add_argument("--punct-dim", type=int, default=2, help="dimension for separator punctuation embeddings")
        group.add_argument("--gap-dim", type=int, default=2, help="dimension for gap type embeddings")
        group.add_argument("--action-dim", type=int, default=5, help="dimension for action type embeddings")
        group.add_argument("--layer-dim", type=int, default=500, help="dimension for hidden layers")
        group.add_argument("--layers", type=int, default=2, help="number of hidden layers")
        group.add_argument("--lstm-layer-dim", type=int, default=500, help="dimension for LSTM hidden layers")
        group.add_argument("--lstm-layers", type=int, default=2, help="number of LSTM hidden layers")
        group.add_argument("--embedding-layer-dim", type=int, default=500, help="dimension for layers before LSTM")
        group.add_argument("--embedding-layers", type=int, default=1, help="number of layers before LSTM")
        group.add_argument("--activation", choices=ACTIVATIONS, default=ACTIVATIONS[0], help="activation function")
        group.add_argument("--init", choices=INITIALIZATIONS, default=INITIALIZATIONS[0], help="weight initialization")
        group.add_argument("--max-labels", type=int, default=100, help="max number of actions to allow")
        group.add_argument("--save-every", type=int, help="every this many passages, evaluate on dev and save model")
        group.add_argument("--minibatch-size", type=int, default=200, help="mini-batch size for optimization")
        group.add_argument("--optimizer", choices=OPTIMIZERS, default=OPTIMIZERS[0], help="algorithm for optimization")
        group.add_argument("--max-words-external", type=int, help="max external word vectors to use")
        group.add_argument("--max-words", type=int, default=10000, help="max number of words to keep embeddings for")
        group.add_argument("--max-tags", type=int, default=100, help="max number of POS tags to keep embeddings for")
        group.add_argument("--max-deps", type=int, default=100, help="max number of dep labels to keep embeddings for")
        group.add_argument("--max-edge-labels", type=int, default=15, help="max number of edge labels for embeddings")
        group.add_argument("--max-puncts", type=int, default=5, help="max number of punctuations for embeddings")
        group.add_argument("--max-gaps", type=int, default=3, help="max number of gap types to keep embeddings for")
        group.add_argument("--max-actions", type=int, default=10, help="max number of action types for embeddings")
        group.add_argument("--word-dropout", type=float, default=0.25, help="word dropout parameter")
        group.add_argument("--word-dropout-external", type=float, default=0.25, help="word dropout for word vectors")
        group.add_argument("--dropout", type=float, default=0.5, help="dropout parameter between layers")
        group = argparser.add_argument_group(title="DyNet parameters")
        group.add_argument("--dynet-mem", help="memory for dynet")
        group.add_argument("--dynet-weight-decay", type=float, help="weight decay for parameters (default 1e-6)")
        group.add_argument("--dynet-gpu", action="store_true", help="use the GPU")
        group.add_argument("--dynet-gpus", type=int, help="how many GPUs you want to use")
        group.add_argument("--dynet-gpu-ids", help="the GPUs that you want to use by device ID")
        group.add_argument("--dynet-viz", action="store_true", help="visualize NN and exit")
        self.args = argparser.parse_args(args if args else None)
        
        if self.args.model:
            if not self.args.log:
                self.args.log = self.args.model + ".log"
            if self.args.dev and not self.args.devscores:
                self.args.devscores = self.args.model + ".dev.csv"
            if self.args.passages and not self.args.testscores:
                self.args.testscores = self.args.model + ".test.csv"
        elif not self.args.log:
            self.args.log = "parse.log"

        self._log_file = None
        self.set_external()
        self.random = np.random

    def set_external(self):
        np.random.seed(self.args.seed)
        sys.argv += ["--dynet-seed", str(self.args.seed)]
        if self.args.dynet_mem:
            sys.argv += ["--dynet-mem", str(self.args.dynet_mem)]
        if self.args.dynet_weight_decay:
            sys.argv += ["--dynet-weight-decay", str(self.args.dynet_weight_decay)]
        if self.args.dynet_gpu:
            sys.argv += ["--dynet-gpu"]
        if self.args.dynet_gpus:
            sys.argv += ["--dynet-gpus", str(self.args.dynet_gpus)]
        if self.args.dynet_gpu_ids:
            sys.argv += ["--dynet-gpu-ids", str(self.args.dynet_gpu_ids)]
        if self.args.dynet_viz:
            sys.argv += ["--dynet-viz"]

    def update(self, params):
        for name, value in params.items():
            setattr(self.args, name, value)
        self.set_external()

    @property
    def line_end(self):
        return "\n" if self.args.verbose else " "  # show all in one line unless verbose

    def log(self, message):
        try:
            if self._log_file is None:
                self._log_file = open(self.args.log, "w")
            print(message, file=self._log_file, flush=True)
            if self.args.verbose:
                print(message)
        except OSError:
            pass

    def close(self):
        if self._log_file is not None:
            self._log_file.close()

    def __str__(self):
        return " ".join("%s=%s" % item for item in sorted(vars(self.args).items()))
