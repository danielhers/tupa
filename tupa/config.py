import sys
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

import numpy as np
from logbook import Logger, FileHandler, StderrHandler
from ucca import constructions

from scheme.cfgutil import Singleton, add_verbose_argument, add_boolean_option, get_group_arg_names
from scheme.convert import UCCA_EXT, CONVERTERS
from tupa.classifiers.nn.constants import *

# Classifiers
SPARSE = "sparse"
MLP_NN = "mlp"
BILSTM_NN = "bilstm"
NOOP = "noop"
NN_CLASSIFIERS = (MLP_NN, BILSTM_NN)
CLASSIFIERS = (SPARSE, MLP_NN, BILSTM_NN, NOOP)

# Swap types
REGULAR = "regular"
COMPOUND = "compound"

# Input/output formats
FORMATS = [e.lstrip(".") for e in UCCA_EXT] + ["ucca"] + list(CONVERTERS)

# Required number of edge labels per format
EDGE_LABELS_NUM = {"amr": 110, "sdp": 70, "conllu": 60}


class Hyperparams(object):
    def __init__(self, shared=None, **kwargs):
        self.shared = shared if isinstance(shared, Namespace) else Namespace(**shared) if shared else Namespace()
        self.specific = {k: v if isinstance(v, Namespace) else Namespace(**v) for k, v in kwargs.items()}


class HyperparamsInitializer(object):
    def __init__(self, name=None, *args, parent=None, **kwargs):
        self.parent = parent
        self.name = name
        self.args = args
        if kwargs:  # parent is a Namespace and kwargs are already parsed and initialized values
            self.parsed_args = vars(parent)
            self.parsed_args.update(kwargs)
        else:  # parent is an ArgumentParser and args are strings
            self.parsed_args = ArgumentParser(parents=[parent], add_help=False).parse_args(args)

    def __call__(self, args):
        return HyperparamsInitializer(*args.replace("=", " ").split(), parent=self.parent)

    def __str__(self):
        return '"%s"' % " ".join([self.name] + list(self.args)) if self.args else self.name


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        argparser = ArgumentParser(description="Transition-based parser for UCCA.",
                                   formatter_class=ArgumentDefaultsHelpFormatter)
        argparser.add_argument("passages", nargs="*", help="passage files/directories to test on/parse")
        argparser.add_argument("-m", "--model", help="model file basename to load/save (default: <format>_<model_type>")
        argparser.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=SPARSE, help="model type")
        argparser.add_argument("-B", "--beam", type=int, choices=(1,), default=1, help="beam size for beam search")
        add_boolean_option(argparser, "evaluate", "evaluation of parsed passages", short="e")
        add_verbose_argument(argparser, help="detailed parse output")
        group = argparser.add_argument_group(title="Node labels")
        group.add_argument("--max-node-labels", type=int, default=0, help="max number of node labels to allow")
        group.add_argument("--max-node-categories", type=int, default=0, help="max number of node categories to allow")
        group.add_argument("--min-node-label-count", type=int, default=2, help="min number of occurrences for a label")
        add_boolean_option(group, "use-gold-node-labels", "gold node labels when parsing")
        add_boolean_option(group, "wikification", "use Spotlight to wikify any named node")
        constructions.add_argument(argparser)
        add_boolean_option(argparser, "sentences", "split to sentences")
        add_boolean_option(argparser, "paragraphs", "split to paragraphs")
        group = argparser.add_argument_group(title="Training parameters")
        group.add_argument("-t", "--train", nargs="+", default=(), help="passage files/directories to train on")
        group.add_argument("-d", "--dev", nargs="+", default=(), help="passage files/directories to tune on")
        group.add_argument("-I", "--iterations", type=int, default=1, help="number of training iterations")
        group.add_argument("--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        group.add_argument("--seed", type=int, default=1, help="random number generator seed")
        group = argparser.add_argument_group(title="Output files")
        group.add_argument("-o", "--outdir", default=".", help="output directory for parsed files")
        group.add_argument("-p", "--prefix", default="", help="output filename prefix")
        add_boolean_option(group, "write", "writing parsed output to files", default=True, short_no="W")
        group.add_argument("-l", "--log", help="output log file (default: model filename + .log)")
        group.add_argument("--devscores", help="output CSV file for dev scores (default: model filename + .dev.csv)")
        group.add_argument("--testscores", help="output CSV file for test scores (default: model filename + .test.csv)")
        argparser.add_argument("-f", "--formats", nargs="*", choices=FORMATS,
                               help="input formats for creating all parameters before training starts "
                                    "(otherwise created dynamically based on filename suffix), "
                                    "and output formats for written files (each will be written; default: UCCA XML)")
        add_boolean_option(group, "node-labels", "prediction of node labels, if supported by format", default=True)
        group = argparser.add_argument_group(title="Structural constraints")
        add_boolean_option(group, "linkage", "linkage nodes and edges")
        add_boolean_option(group, "implicit", "implicit nodes and edges")
        add_boolean_option(group, "remote", "remote edges", default=True)
        add_boolean_option(group, "constraints", "scheme-specific rules", default=True)
        add_boolean_option(group, "require-connected", "constraint that output graph must be connected")
        group.add_argument("--orphan-label", default="orphan", help="edge label to use for nodes without parents")
        group.add_argument("--max-action-ratio", type=float, default=100, help="max action/terminal ratio")
        group.add_argument("--max-node-ratio", type=float, default=10, help="max node/terminal ratio")
        group.add_argument("--max-height", type=int, default=20, help="max graph height")
        group = argparser.add_mutually_exclusive_group()
        group.add_argument("--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
        group.add_argument("--no-swap", action="store_false", dest="swap", help="exclude swap transitions")
        argparser.add_argument("--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")
        group = argparser.add_argument_group(title="Sanity checks")
        add_boolean_option(group, "check-loops", "check for parser state loop")
        add_boolean_option(group, "verify", "check for oracle reproducing original passage")
        group = argparser.add_argument_group(title="General classifier training parameters")
        group.add_argument("--learning-rate", type=float, help="rate for model weight updates (default: by trainer/1)")
        group.add_argument("--learning-rate-decay", type=float, default=0.0, help="learning rate decay per iteration")
        group.add_argument("--swap-importance", type=float, default=1, help="learning rate factor for Swap")
        add_boolean_option(group, "early-update", "early update procedure (finish example on first error)")
        group.add_argument("--save-every", type=int, help="every this many passages, evaluate on dev and save model")
        group = argparser.add_argument_group(title="Perceptron parameters")
        group.add_argument("--min-update", type=int, default=5, help="minimum #updates for using a feature")
        self.sparse_arg_names = get_group_arg_names(group)
        group = argparser.add_argument_group(title="Neural network parameters")
        group.add_argument("--word-dim-external", type=int, default=300, help="dimension for external word embeddings")
        group.add_argument("--word-vectors", help="file to load external word embeddings from (default: GloVe)")
        add_boolean_option(group, "update-word-vectors", "external word vectors in training parameters", default=True)
        group.add_argument("--word-dim", type=int, default=200, help="dimension for learned word embeddings")
        group.add_argument("--tag-dim", type=int, default=20, help="dimension for POS tag embeddings")
        group.add_argument("--dep-dim", type=int, default=10, help="dimension for dependency relation embeddings")
        group.add_argument("--edge-label-dim", type=int, default=20, help="dimension for edge label embeddings")
        group.add_argument("--node-label-dim", type=int, default=0, help="dimension for node label embeddings")
        group.add_argument("--node-category-dim", type=int, default=0, help="dimension for node category embeddings")
        group.add_argument("--punct-dim", type=int, default=1, help="dimension for separator punctuation embeddings")
        group.add_argument("--action-dim", type=int, default=3, help="dimension for input action type embeddings")
        group.add_argument("--ner-dim", type=int, default=5, help="dimension for input entity type embeddings")
        group.add_argument("--output-dim", type=int, default=50, help="dimension for output action embeddings")
        group.add_argument("--layer-dim", type=int, default=50, help="dimension for hidden layers")
        group.add_argument("--layers", type=int, default=2, help="number of hidden layers")
        group.add_argument("--lstm-layer-dim", type=int, default=500, help="dimension for LSTM hidden layers")
        group.add_argument("--lstm-layers", type=int, default=0, help="number of LSTM hidden layers")
        group.add_argument("--embedding-layer-dim", type=int, default=500, help="dimension for layers before LSTM")
        group.add_argument("--embedding-layers", type=int, default=1, help="number of layers before LSTM")
        group.add_argument("--activation", choices=ACTIVATIONS, default=DEFAULT_ACTIVATION, help="activation function")
        group.add_argument("--init", choices=INITIALIZERS, default=DEFAULT_INITIALIZER, help="weight initialization")
        group.add_argument("--minibatch-size", type=int, default=100, help="mini-batch size for optimization")
        group.add_argument("--optimizer", choices=TRAINERS, default=DEFAULT_TRAINER, help="algorithm for optimization")
        group.add_argument("--max-words-external", type=int, help="max external word vectors to use")
        group.add_argument("--max-words", type=int, default=10000, help="max number of words to keep embeddings for")
        group.add_argument("--max-tags", type=int, default=100, help="max number of POS tags to keep embeddings for")
        group.add_argument("--max-deps", type=int, default=100, help="max number of dep labels to keep embeddings for")
        group.add_argument("--max-edge-labels", type=int, default=15, help="max number of edge labels for embeddings")
        group.add_argument("--max-puncts", type=int, default=5, help="max number of punctuations for embeddings")
        group.add_argument("--max-action-types", type=int, default=10, help="max number of action types for embeddings")
        group.add_argument("--max-action-labels", type=int, default=100, help="max number of action labels to allow")
        group.add_argument("--max-ner-types", type=int, default=18, help="max number of entity types to allow")
        group.add_argument("--word-dropout", type=float, default=0.2, help="word dropout parameter")
        group.add_argument("--word-dropout-external", type=float, default=0, help="word dropout for word vectors")
        group.add_argument("--tag-dropout", type=float, default=0.2, help="POS tag dropout parameter")
        group.add_argument("--dep-dropout", type=float, default=0.2, help="dependency label dropout parameter")
        group.add_argument("--node-label-dropout", type=float, default=0.2, help="node label dropout parameter")
        group.add_argument("--dropout", type=float, default=0.4, help="dropout parameter between layers")
        group.add_argument("--max-length", type=int, default=120, help="maximum length of input sentence")
        group.add_argument("--rnn", choices=RNNS, default=DEFAULT_RNN, help="type of recurrent neural network to use")
        self.nn_arg_names = get_group_arg_names(group)
        group = argparser.add_argument_group(title="DyNet parameters")
        group.add_argument("--dynet-mem", help="memory for dynet")
        group.add_argument("--dynet-weight-decay", type=float, default=1e-5, help="weight decay for parameters")
        add_boolean_option(group, "dynet-gpu", "GPU for training")
        group.add_argument("--dynet-gpus", type=int, default=1, help="how many GPUs you want to use")
        group.add_argument("--dynet-gpu-ids", help="the GPUs that you want to use by device ID")
        group.add_argument("--dynet-devices")
        add_boolean_option(group, "dynet-viz", "visualization of neural network structure")
        add_boolean_option(group, "dynet-autobatch", "auto-batching of training examples")
        initializer = HyperparamsInitializer(parent=argparser)
        argparser.add_argument("-H", "--hyperparams", type=initializer, nargs="*",
                               help="shared hyperparameters or hyperparameters for specific formats, "
                                    'e.g., "shared --lstm-layer-dim=100 --lstm-layers=1" "ucca --word-dim=300"',
                               default=[initializer("shared --lstm-layers 2")])
        self.dynet_arg_names = get_group_arg_names(group)
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
        self._logger = self.format = None
        self.original_values = self.create_original_values()
        self.hyperparams = self.create_hyperparams()
        self.set_format()
        self.set_external()
        self.random = np.random

    def create_original_values(self, args=None):
        return {attr: getattr(self.args, attr) if args is None else args[attr]
                for attr in ("node_labels", "implicit", "node_label_dim", "node_category_dim", "max_node_labels",
                             "max_node_categories", "max_action_labels", "max_edge_labels")
                if args is None or attr in args}

    def create_hyperparams(self):
        return Hyperparams(**{h.name: h.parsed_args for h in self.args.hyperparams or ()})

    def set_format(self, f=None):
        if self.format != f:
            format_values = dict(self.original_values)
            format_values.update(vars(self.hyperparams.specific.get(self.format, Namespace())))
            for attr, value in format_values.items():
                setattr(self.args, attr, value)
        if f not in (None, "text"):
            self.format = f
        if self.format == "amr":
            self.args.implicit = True
            if not self.args.node_label_dim:
                self.args.node_label_dim = 20
            if not self.args.max_node_labels:
                self.args.max_node_labels = 1000
            if not self.args.node_category_dim:
                self.args.node_category_dim = 5
            if not self.args.max_node_categories:
                self.args.max_node_categories = 25
        else:  # All other formats do not use node labels
            self.args.node_labels = False
            self.args.node_label_dim = self.args.max_node_labels = \
                self.args.node_category_dim = self.args.max_node_categories = 0
        required_edge_labels = EDGE_LABELS_NUM.get(self.format)
        if required_edge_labels is not None:
            self.args.max_edge_labels = max(self.args.max_edge_labels, required_edge_labels)
            self.args.max_action_labels = max(self.args.max_action_labels, 6 * required_edge_labels)

    def set_external(self):
        np.random.seed(self.args.seed)
        sys.argv += ["--dynet-seed", str(self.args.seed)]
        if self.args.dynet_mem:
            sys.argv += ["--dynet-mem", str(self.args.dynet_mem)]
        if self.args.dynet_weight_decay:
            sys.argv += ["--dynet-weight-decay", str(self.args.dynet_weight_decay)]
        if self.args.dynet_gpu:
            sys.argv += ["--dynet-gpu"]
        if self.args.dynet_gpus and self.args.dynet_gpus != 1:
            sys.argv += ["--dynet-gpus", str(self.args.dynet_gpus)]
        if self.args.dynet_gpu_ids:
            sys.argv += ["--dynet-gpu-ids", str(self.args.dynet_gpu_ids)]
        if self.args.dynet_devices:
            sys.argv += ["--dynet-devices", str(self.args.dynet_devices)]
        if self.args.dynet_viz:
            sys.argv += ["--dynet-viz"]
        if self.args.dynet_autobatch:
            sys.argv += ["--dynet-autobatch", "1"]

    def update(self, params):
        for name, value in params.items():
            setattr(self.args, name, value)
        self.original_values.update(self.create_original_values(params))
        self.hyperparams = self.create_hyperparams()
        self.set_format()
        self.set_external()

    def update_hyperparams(self, **kwargs):
        self.update({"hyperparams": [HyperparamsInitializer(k, parent=self.args, **v) for k, v in kwargs.items()]})

    @property
    def line_end(self):
        return "\n" if self.args.verbose > 2 else " "  # show all in one line unless verbose

    def log(self, message):
        try:
            if self._logger is None:
                FileHandler(self.args.log,
                            format_string="{record.time:%Y-%m-%d %H:%M:%S} {record.message}").push_application()
                if self.args.verbose > 1:
                    StderrHandler(bubble=True).push_application()
                self._logger = Logger("tupa")
            self._logger.warn(message)
        except OSError:
            pass

    def __str__(self):
        return " ".join(list(self.args.passages) + [""]) + \
               " ".join("--" + ("no-" if v is False else "") + k.replace("_", "-") +
                        ("" if v is False or v is True else
                         (" " + str(" ".join(map(str, v)) if hasattr(v, "__iter__") and not isinstance(v, str) else v)))
                        for (k, v) in sorted(vars(self.args).items()) if
                        v not in (None, (), "")
                        and not k.startswith("_")
                        and (self.args.node_labels or ("node_label" not in k and "node_categor" not in k))
                        and (self.args.swap or "swap_" not in k)
                        and (self.args.swap == COMPOUND or k != "max_swap")
                        and (not self.args.require_connected or k != "orphan_label")
                        and (self.args.classifier == SPARSE or k not in self.sparse_arg_names)
                        and (
                            self.args.classifier in NN_CLASSIFIERS or k not in self.nn_arg_names + self.dynet_arg_names)
                        and k != "passages")
