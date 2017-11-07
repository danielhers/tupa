import sys
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter, SUPPRESS
from collections import defaultdict
from functools import partial

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
SPARSE_ARG_NAMES = set()
NN_ARG_NAMES = set()
DYNET_ARG_NAMES = set()
RESTORED_ARGS = set()


def add_param_arguments(argparser=None, arg_default=None):  # arguments with possible format-specific parameter values
    
    def add_argument(a, *args, **kwargs):
        return a.add_argument(*args, **kwargs)
    
    def add(a, *args, default=None, restore=False, func=add_argument, **kwargs):
        arg = func(a, *args, default=default if arg_default is None else arg_default, **kwargs)
        if restore:
            try:
                RESTORED_ARGS.add(arg.dest)
            except AttributeError:
                RESTORED_ARGS.update(get_group_arg_names(arg))
    
    def add_boolean(a, *args, **kwargs):
        add(a, *args, func=add_boolean_option, **kwargs)

    if not argparser:
        argparser = ArgumentParser()
    
    group = argparser.add_argument_group(title="Node labels")
    add(group, "--max-node-labels", type=int, default=0, help="max number of node labels to allow", restore=True)
    add(group, "--max-node-categories", type=int, default=0, help="max node categories to allow", restore=True)
    add(group, "--min-node-label-count", type=int, default=2, help="min number of occurrences for a label")
    add_boolean(group, "use-gold-node-labels", "gold node labels when parsing")
    add_boolean(group, "wikification", "use Spotlight to wikify any named node")
    add_boolean(group, "node-labels", "prediction of node labels, if supported by format", default=True, restore=True)

    group = argparser.add_argument_group(title="Structural constraints")
    add_boolean(group, "linkage", "linkage nodes and edges")
    add_boolean(group, "implicit", "implicit nodes and edges", restore=True)
    add_boolean(group, "remote", "remote edges", default=True)
    add_boolean(group, "constraints", "scheme-specific rules", default=True)
    add_boolean(group, "require-connected", "constraint that output graph must be connected")
    add(group, "--orphan-label", default="orphan", help="edge label to use for nodes without parents")
    add(group, "--max-action-ratio", type=float, default=100, help="max action/terminal ratio", restore=True)
    add(group, "--max-node-ratio", type=float, default=10, help="max node/terminal ratio")
    add(group, "--max-height", type=int, default=20, help="max graph height")

    group = argparser.add_mutually_exclusive_group()
    add(group, "--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
    add(group, "--no-swap", action="store_false", dest="swap", help="exclude swap transitions")
    add(argparser, "--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")

    group = argparser.add_argument_group(title="General classifier training parameters")
    add(group, "--learning-rate", type=float, help="rate for model weight updates (default: by trainer/1)")
    add(group, "--learning-rate-decay", type=float, default=0.0, help="learning rate decay per iteration")
    add(group, "--swap-importance", type=float, default=1, help="learning rate factor for Swap")

    group = argparser.add_argument_group(title="Perceptron parameters")
    add(group, "--min-update", type=int, default=5, help="minimum #updates for using a feature")
    SPARSE_ARG_NAMES.update(get_group_arg_names(group))

    group = argparser.add_argument_group(title="Neural network parameters")
    add(group, "--word-dim-external", type=int, default=300, help="dimension for external word embeddings")
    add(group, "--word-vectors", help="file to load external word embeddings from (default: GloVe)")
    add_boolean(group, "update-word-vectors", "external word vectors in training parameters", default=True)
    add(group, "--word-dim", type=int, default=200, help="dimension for learned word embeddings")
    add(group, "--tag-dim", type=int, default=20, help="dimension for POS tag embeddings")
    add(group, "--dep-dim", type=int, default=10, help="dimension for dependency relation embeddings")
    add(group, "--edge-label-dim", type=int, default=20, help="dimension for edge label embeddings")
    add(group, "--node-label-dim", type=int, default=0, help="dimension for node label embeddings", restore=True)
    add(group, "--node-category-dim", type=int, default=0, help="dimension for node category embeddings", restore=True)
    add(group, "--punct-dim", type=int, default=1, help="dimension for separator punctuation embeddings")
    add(group, "--action-dim", type=int, default=3, help="dimension for input action type embeddings")
    add(group, "--ner-dim", type=int, default=5, help="dimension for input entity type embeddings")
    add(group, "--output-dim", type=int, default=50, help="dimension for output action embeddings")
    add(group, "--layer-dim", type=int, default=50, help="dimension for hidden layers")
    add(group, "--layers", type=int, default=2, help="number of hidden layers")
    add(group, "--lstm-layer-dim", type=int, default=500, help="dimension for LSTM hidden layers")
    add(group, "--lstm-layers", type=int, default=0, help="number of LSTM hidden layers")
    add(group, "--embedding-layer-dim", type=int, default=500, help="dimension for layers before LSTM")
    add(group, "--embedding-layers", type=int, default=1, help="number of layers before LSTM")
    add(group, "--activation", choices=ACTIVATIONS, default=DEFAULT_ACTIVATION, help="activation function")
    add(group, "--init", choices=INITIALIZERS, default=DEFAULT_INITIALIZER, help="weight initialization")
    add(group, "--minibatch-size", type=int, default=100, help="mini-batch size for optimization")
    add(group, "--optimizer", choices=TRAINERS, default=DEFAULT_TRAINER, help="algorithm for optimization")
    add(group, "--max-words-external", type=int, help="max external word vectors to use")
    add(group, "--max-words", type=int, default=10000, help="max number of words to keep embeddings for")
    add(group, "--max-tags", type=int, default=100, help="max number of POS tags to keep embeddings for")
    add(group, "--max-deps", type=int, default=100, help="max number of dep labels to keep embeddings for")
    add(group, "--max-edge-labels", type=int, default=15, help="max number of edge labels for embeddings", restore=True)
    add(group, "--max-puncts", type=int, default=5, help="max number of punctuations for embeddings")
    add(group, "--max-action-types", type=int, default=10, help="max number of action types for embeddings")
    add(group, "--max-action-labels", type=int, default=100, help="max number of action labels to allow")
    add(group, "--max-ner-types", type=int, default=18, help="max number of entity types to allow")
    add(group, "--word-dropout", type=float, default=0.2, help="word dropout parameter")
    add(group, "--word-dropout-external", type=float, default=0, help="word dropout for word vectors")
    add(group, "--tag-dropout", type=float, default=0.2, help="POS tag dropout parameter")
    add(group, "--dep-dropout", type=float, default=0.2, help="dependency label dropout parameter")
    add(group, "--node-label-dropout", type=float, default=0.2, help="node label dropout parameter")
    add(group, "--dropout", type=float, default=0.4, help="dropout parameter between layers")
    add(group, "--max-length", type=int, default=120, help="maximum length of input sentence")
    add(group, "--rnn", choices=RNNS, default=DEFAULT_RNN, help="type of recurrent neural network to use")
    NN_ARG_NAMES.update(get_group_arg_names(group))

    return argparser


class FallbackNamespace(Namespace):
    def __init__(self, fallback, kwargs=None):
        super().__init__(**(kwargs or {}))
        self._fallback = fallback

    def __getattr__(self, item):
        return getattr(super(), item, getattr(self._fallback, item))


class Hyperparams(object):
    def __init__(self, parent, shared=None, **kwargs):
        self.shared = FallbackNamespace(parent, shared)
        self.specific = defaultdict(partial(FallbackNamespace, parent),
                                    **{k: FallbackNamespace(parent, v) for k, v in kwargs.items()})


class HyperparamsInitializer(object):
    def __init__(self, name=None, *args, **kwargs):
        """
        :param name: name of hyperparams subset
        :param args: raw arg strings
        :param kwargs: parsed and initialized values
        """
        self.name = name
        self.str_args = list(args) + ["--%s %s" % k for k in kwargs.items()]
        self.args = vars(add_param_arguments(arg_default=SUPPRESS).parse_args(args))
        self.args.update(kwargs)

    def __str__(self):
        return '"%s"' % " ".join([self.name] + list(self.str_args))


def hyperparams_action(args):
    return HyperparamsInitializer(*args.replace("=", " ").split())


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
        constructions.add_argument(argparser)
        add_boolean_option(argparser, "sentences", "split to sentences")
        add_boolean_option(argparser, "paragraphs", "split to paragraphs")
        argparser.add_argument("--timeout", type=float, help="max number of seconds to wait for a single passage")

        group = argparser.add_argument_group(title="Training parameters")
        group.add_argument("-t", "--train", nargs="+", default=(), help="passage files/directories to train on")
        group.add_argument("-d", "--dev", nargs="+", default=(), help="passage files/directories to tune on")
        group.add_argument("-I", "--iterations", type=int, default=1, help="number of training iterations")
        group.add_argument("--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        group.add_argument("--seed", type=int, default=1, help="random number generator seed")
        add_boolean_option(group, "early-update", "early update procedure (finish example on first error)")
        group.add_argument("--save-every", type=int, help="every this many passages, evaluate on dev and save model")

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

        group = argparser.add_argument_group(title="Sanity checks")
        add_boolean_option(group, "check-loops", "check for parser state loop")
        add_boolean_option(group, "verify", "check for oracle reproducing original passage")
        add_param_arguments(argparser)

        group = argparser.add_argument_group(title="DyNet parameters")
        group.add_argument("--dynet-mem", help="memory for dynet")
        group.add_argument("--dynet-weight-decay", type=float, default=1e-5, help="weight decay for parameters")
        add_boolean_option(group, "dynet-gpu", "GPU for training")
        group.add_argument("--dynet-gpus", type=int, default=1, help="how many GPUs you want to use")
        group.add_argument("--dynet-gpu-ids", help="the GPUs that you want to use by device ID")
        group.add_argument("--dynet-devices")
        add_boolean_option(group, "dynet-viz", "visualization of neural network structure")
        add_boolean_option(group, "dynet-autobatch", "auto-batching of training examples")
        DYNET_ARG_NAMES.update(get_group_arg_names(group))

        argparser.add_argument("-H", "--hyperparams", type=hyperparams_action, nargs="*",
                               help="shared hyperparameters or hyperparameters for specific formats, "
                                    'e.g., "shared --lstm-layer-dim=100 --lstm-layers=1" "ucca --word-dim=300"',
                               default=[hyperparams_action("shared --lstm-layers 2")])

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
        self._logger = self.format = self.hyperparams = None
        self.original_values = {}
        self.random = np.random
        self.update()

    def create_original_values(self, args=None):
        return {attr: getattr(self.args, attr) if args is None else args[attr]
                for attr in RESTORED_ARGS if args is None or attr in args}

    def set_format(self, f=None):
        if self.format != f:
            format_values = dict(self.original_values)
            format_values.update({k: v for k, v in vars(self.hyperparams.specific[self.format]).items()
                                  if not k.startswith("_")})
            for attr, value in format_values.items():
                setattr(self.args, attr, value)
        if f in (None, "text"):
            if not self.format:
                self.format = "ucca"
        else:
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

    def set_dynet_arguments(self):
        self.random.seed(self.args.seed)
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

    def update(self, params=None):
        if params:
            for name, value in params.items():
                setattr(self.args, name, value)
        self.original_values.update(self.create_original_values(params))
        self.hyperparams = self.create_hyperparams()
        self.set_format()
        self.set_dynet_arguments()

    def create_hyperparams(self):
        return Hyperparams(parent=self.args, **{h.name: h.args for h in self.args.hyperparams or ()})

    def update_hyperparams(self, **kwargs):
        self.update({"hyperparams": [HyperparamsInitializer(k, **v) for k, v in kwargs.items()]})

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
                        and (self.args.classifier == SPARSE or k not in SPARSE_ARG_NAMES)
                        and (
                            self.args.classifier in NN_CLASSIFIERS or k not in NN_ARG_NAMES | DYNET_ARG_NAMES)
                        and k != "passages")
