import os
import shlex
from copy import deepcopy

import dynet_config
import numpy as np
from configargparse import ArgParser, Namespace, ArgumentDefaultsHelpFormatter, SUPPRESS
from logbook import Logger, FileHandler, StderrHandler
from semstr.cfgutil import Singleton, add_verbose_arg, add_boolean_option, get_group_arg_names
from semstr.convert import UCCA_EXT, CONVERTERS
from ucca import constructions

from tupa.classifiers.nn.constants import *
from tupa.model_util import load_enum

# Classifiers

SPARSE = "sparse"
MLP = "mlp"
BIRNN = "bilstm"
HIGHWAY_RNN = "highway"
HIERARCHICAL_RNN = "hbirnn"
NOOP = "noop"
NN_CLASSIFIERS = (MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN)
CLASSIFIERS = (SPARSE, MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN, NOOP)

FEATURE_PROPERTIES = "wmtudhencpqxyAPCIEMNT#^$"

# Swap types
REGULAR = "regular"
COMPOUND = "compound"

# Input/output formats
FORMATS = ["ucca"] + list(CONVERTERS)
FILE_FORMATS = [e.lstrip(".") for e in UCCA_EXT] + FORMATS

# Required number of edge labels per format
EDGE_LABELS_NUM = {"amr": 110, "sdp": 70, "conllu": 60}
SPARSE_ARG_NAMES = set()
NN_ARG_NAMES = set()
DYNET_ARG_NAMES = set()
RESTORED_ARGS = set()

SEPARATOR = "."


def add_param_arguments(ap=None, arg_default=None):  # arguments with possible format-specific parameter values

    def add_argument(a, *args, **kwargs):
        return a.add_argument(*args, **kwargs)

    def add(a, *args, default=None, func=add_argument, **kwargs):
        arg = func(a, *args, default=default if arg_default is None else arg_default, **kwargs)
        try:
            RESTORED_ARGS.add(arg.dest)
        except AttributeError:
            RESTORED_ARGS.update(get_group_arg_names(arg))

    def add_boolean(a, *args, **kwargs):
        add(a, *args, func=add_boolean_option, **kwargs)

    if not ap:
        ap = ArgParser()

    group = ap.add_argument_group(title="Node labels")
    add(group, "--max-node-labels", type=int, default=0, help="max number of node labels to allow")
    add(group, "--max-node-categories", type=int, default=0, help="max node categories to allow")
    add(group, "--min-node-label-count", type=int, default=2, help="min number of occurrences for a label")
    add_boolean(group, "use-gold-node-labels", "gold node labels when parsing")
    add_boolean(group, "wikification", "use Spotlight to wikify any named node")
    add_boolean(group, "node-labels", "prediction of node labels, if supported by format", default=True)

    group = ap.add_argument_group(title="Structural constraints")
    add_boolean(group, "linkage", "linkage nodes and edges")
    add_boolean(group, "implicit", "implicit nodes and edges")
    add_boolean(group, "remote", "remote edges", default=True)
    add_boolean(group, "constraints", "scheme-specific rules", default=True)
    add_boolean(group, "require-connected", "constraint that output graph must be connected")
    add(group, "--orphan-label", default="orphan", help="edge label to use for nodes without parents")
    add(group, "--max-action-ratio", type=float, default=100, help="max action/terminal ratio")
    add(group, "--max-node-ratio", type=float, default=10, help="max node/terminal ratio")
    add(group, "--max-height", type=int, default=20, help="max graph height")

    group = ap.add_mutually_exclusive_group()
    add(group, "--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
    add(group, "--no-swap", action="store_false", dest="swap", help="exclude swap transitions")
    add(ap, "--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")

    group = ap.add_argument_group(title="General classifier training parameters")
    add(group, "--learning-rate", type=float, help="rate for model weight updates (default: by trainer/1)")
    add(group, "--learning-rate-decay", type=float, default=0, help="learning rate decay per iteration")
    add(group, "--swap-importance", type=float, default=1, help="learning rate factor for Swap")
    add(group, "--max-training-per-format", type=int, help="max number of training passages per format per iteration")
    add_boolean(group, "missing-node-features", "allow node features to be missing if not available", default=True)
    add(group, "--omit-features", help="string of feature properties to omit, out of " + FEATURE_PROPERTIES)

    group = ap.add_argument_group(title="Perceptron parameters")
    add(group, "--min-update", type=int, default=5, help="minimum #updates for using a feature")
    SPARSE_ARG_NAMES.update(get_group_arg_names(group))

    group = ap.add_argument_group(title="Neural network parameters")
    add(group, "--word-dim-external", type=int, default=300, help="dimension for external word embeddings")
    add(group, "--word-vectors", help="file to load external word embeddings from (default: GloVe)")
    add(group, "--vocab", help="file mapping integer ID to word form (to avoid loading spaCy), or '-' to use word form")
    add_boolean(group, "update-word-vectors", "external word vectors in training parameters", default=True)
    add(group, "--word-dim", type=int, default=0, help="dimension for learned word embeddings")
    add(group, "--lemma-dim", type=int, default=200, help="dimension for lemma embeddings")
    add(group, "--tag-dim", type=int, default=20, help="dimension for fine POS tag embeddings")
    add(group, "--pos-dim", type=int, default=20, help="dimension for coarse/universal POS tag embeddings")
    add(group, "--dep-dim", type=int, default=10, help="dimension for dependency relation embeddings")
    add(group, "--edge-label-dim", type=int, default=20, help="dimension for edge label embeddings")
    add(group, "--node-label-dim", type=int, default=0, help="dimension for node label embeddings")
    add(group, "--node-category-dim", type=int, default=0, help="dimension for node category embeddings")
    add(group, "--punct-dim", type=int, default=1, help="dimension for separator punctuation embeddings")
    add(group, "--action-dim", type=int, default=3, help="dimension for input action type embeddings")
    add(group, "--ner-dim", type=int, default=3, help="dimension for input entity type embeddings")
    add(group, "--shape-dim", type=int, default=3, help="dimension for word shape embeddings")
    add(group, "--prefix-dim", type=int, default=2, help="dimension for word prefix embeddings")
    add(group, "--suffix-dim", type=int, default=3, help="dimension for word suffix embeddings")
    add(group, "--output-dim", type=int, default=50, help="dimension for output action embeddings")
    add(group, "--layer-dim", type=int, default=50, help="dimension for hidden layers")
    add(group, "--layers", type=int, default=2, help="number of hidden layers")
    add(group, "--lstm-layer-dim", type=int, default=500, help="dimension for LSTM hidden layers")
    add(group, "--lstm-layers", type=int, default=0, help="number of LSTM hidden layers")
    add(group, "--embedding-layer-dim", type=int, default=500, help="dimension for layers before LSTM")
    add(group, "--embedding-layers", type=int, default=1, help="number of layers before LSTM")
    add(group, "--activation", choices=ACTIVATIONS, default=DEFAULT_ACTIVATION, help="activation function")
    add(group, "--init", choices=INITIALIZERS, default=DEFAULT_INITIALIZER, help="weight initialization")
    add(group, "--minibatch-size", type=int, default=200, help="mini-batch size for optimization")
    add(group, "--optimizer", choices=TRAINERS, default=DEFAULT_TRAINER, help="algorithm for optimization")
    add(group, "--loss", choices=LOSSES, default=DEFAULT_LOSS, help="loss function for training")
    add(group, "--max-words-external", type=int, default=250000, help="max external word vectors to use")
    add(group, "--max-words", type=int, default=10000, help="max number of words to keep embeddings for")
    add(group, "--max-lemmas", type=int, default=3000, help="max number of lemmas to keep embeddings for")
    add(group, "--max-tags", type=int, default=100, help="max number of fine POS tags to keep embeddings for")
    add(group, "--max-pos", type=int, default=100, help="max number of coarse POS tags to keep embeddings for")
    add(group, "--max-deps", type=int, default=100, help="max number of dep labels to keep embeddings for")
    add(group, "--max-edge-labels", type=int, default=15, help="max number of edge labels for embeddings")
    add(group, "--max-puncts", type=int, default=5, help="max number of punctuations for embeddings")
    add(group, "--max-action-types", type=int, default=10, help="max number of action types for embeddings")
    add(group, "--max-action-labels", type=int, default=100, help="max number of action labels to allow")
    add(group, "--max-ner-types", type=int, default=18, help="max number of entity types to allow")
    add(group, "--max-shapes", type=int, default=30, help="max number of word shapes to allow")
    add(group, "--max-prefixes", type=int, default=30, help="max number of 1-character word prefixes to allow")
    add(group, "--max-suffixes", type=int, default=500, help="max number of 3-character word suffixes to allow")
    add(group, "--word-dropout", type=float, default=0.2, help="word dropout parameter")
    add(group, "--word-dropout-external", type=float, default=0, help="word dropout for word vectors")
    add(group, "--lemma-dropout", type=float, default=0.2, help="lemma dropout parameter")
    add(group, "--tag-dropout", type=float, default=0.2, help="fine POS tag dropout parameter")
    add(group, "--pos-dropout", type=float, default=0.2, help="coarse POS tag dropout parameter")
    add(group, "--dep-dropout", type=float, default=0.5, help="dependency label dropout parameter")
    add(group, "--node-label-dropout", type=float, default=0.2, help="node label dropout parameter")
    add(group, "--node-dropout", type=float, default=0.1, help="probability to drop features for a whole node")
    add(group, "--dropout", type=float, default=0.4, help="dropout parameter between layers")
    add(group, "--max-length", type=int, default=120, help="maximum length of input sentence")
    add(group, "--rnn", choices=["None"] + list(RNNS), default=DEFAULT_RNN, help="type of recurrent neural network")
    add(group, "--gated", type=int, nargs="?", default=0, help="gated input to BiRNN and MLP")
    NN_ARG_NAMES.update(get_group_arg_names(group))
    return ap


class FallbackNamespace(Namespace):
    def __init__(self, fallback, kwargs=None):
        super().__init__(**(kwargs or {}))
        self._fallback = fallback
        self._children = {}

    def __getattr__(self, item):
        if item.startswith("_"):
            return getattr(super(), item)
        return getattr(super(), item, getattr(self._fallback, item))

    def __getitem__(self, item):
        if item:
            name, _, rest = item.partition(SEPARATOR)
            return self._children.setdefault(name, FallbackNamespace(self))[rest]
        return self

    def vars(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def items(self):
        return self.vars().items()

    def update(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def traverse(self, prefix=None):
        if prefix and self.vars():
            yield (prefix, self)
        for name, child in self._children.items():
            yield from child.traverse(SEPARATOR.join(filter(None, (prefix, name))))


class Hyperparams:
    def __init__(self, parent, shared=None, **kwargs):
        self.shared = FallbackNamespace(parent, shared)
        self.specific = FallbackNamespace(parent)
        for name, args in kwargs.items():
            self.specific[name].update(args)

    def items(self):
        return ([("shared", self.shared)] if self.shared.vars() else []) + list(self.specific.traverse())


class HyperparamsInitializer:
    def __init__(self, name=None, *args, **kwargs):
        """
        :param name: name of hyperparams subset
        :param args: raw arg strings
        :param kwargs: parsed and initialized values
        """
        self.name = name
        self.str_args = list(args) + ["--%s %s" % (k.replace("_", "-"), v) for k, v in kwargs.items()]
        self.args = vars(add_param_arguments(arg_default=SUPPRESS).parse_args(args))
        self.args.update(kwargs)

    def __str__(self):
        return '"%s"' % " ".join([self.name] + list(self.str_args))

    def __bool__(self):
        return bool(self.str_args)
    
    @classmethod
    def action(cls, args):
        return cls(*args.replace("=", " ").split())


class Iterations:
    def __init__(self, args):
        try:
            epochs, *hyperparams = args.replace("=", " ").split()
        except (AttributeError, ValueError):
            epochs, *hyperparams = args,
        self.epochs, self.hyperparams = int(epochs), HyperparamsInitializer(str(epochs), *hyperparams)

    def __str__(self):
        return str(self.hyperparams or self.epochs)


class Config(object, metaclass=Singleton):
    def __init__(self, *args):
        self.arg_parser = ap = ArgParser(description="Transition-based parser for UCCA.",
                                         formatter_class=ArgumentDefaultsHelpFormatter)
        ap.add_argument("passages", nargs="*", help="passage files/directories to test on/parse")
        ap.add_argument("-C", "--config", is_config_file=True, help="configuration file to get arguments from")
        ap.add_argument("-m", "--models", nargs="+", help="model file basename(s) to load/save, ensemble if >1 "
                                                          "(default: <format>_<model_type>")
        ap.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=BIRNN, help="model type")
        ap.add_argument("-B", "--beam", type=int, choices=(1,), default=1, help="beam size for beam search")
        add_boolean_option(ap, "evaluate", "evaluation of parsed passages", short="e")
        add_verbose_arg(ap, help="detailed parse output")
        constructions.add_argument(ap)
        add_boolean_option(ap, "sentences", "split to sentences")
        add_boolean_option(ap, "paragraphs", "split to paragraphs")
        ap.add_argument("--timeout", type=float, help="max number of seconds to wait for a single passage")

        group = ap.add_argument_group(title="Training parameters")
        group.add_argument("-t", "--train", nargs="+", default=(), help="passage files/directories to train on")
        group.add_argument("-d", "--dev", nargs="+", default=(), help="passage files/directories to tune on")
        group.add_argument("-I", "--iterations", nargs="+", type=Iterations,
                           default=(Iterations(50), Iterations("100 --optimizer=" + EXTRA_TRAINER)),
                           help="number of training iterations along with optional hyperparameters per part")
        group.add_argument("--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        group.add_argument("--seed", type=int, default=1, help="random number generator seed")
        add_boolean_option(group, "early-update", "early update procedure (finish example on first error)")
        group.add_argument("--save-every", type=int, help="every this many passages, evaluate on dev and save model")
        add_boolean_option(group, "eval-test", "evaluate on test whenever evaluating on dev, but keep results hidden")
        add_boolean_option(group, "ignore-case", "pre-convert all input files to lower-case in training and test")

        group = ap.add_argument_group(title="Output files")
        group.add_argument("-o", "--outdir", default=".", help="output directory for parsed files")
        group.add_argument("-p", "--prefix", default="", help="output filename prefix")
        add_boolean_option(group, "write", "writing parsed output to files", default=True, short_no="W")
        group.add_argument("-j", "--join", help="if output format is textual, write all to one file with this basename")
        group.add_argument("-l", "--log", help="output log file (default: model filename + .log)")
        group.add_argument("--devscores", help="output CSV file for dev scores (default: model filename + .dev.csv)")
        group.add_argument("--testscores", help="output CSV file for test scores (default: model filename + .test.csv)")
        group.add_argument("--action-stats", help="output CSV file for action statistics")
        add_boolean_option(group, "normalize", "apply normalizations to output in case format is UCCA", default=False)
        ap.add_argument("-f", "--formats", nargs="+", choices=FILE_FORMATS, default=(),
                        help="input formats for creating all parameters before training starts "
                             "(otherwise created dynamically based on filename suffix), "
                             "and output formats for written files (each will be written; default: UCCA XML)")
        ap.add_argument("-u", "--unlabeled", nargs="*", choices=FORMATS, help="to ignore labels in")
        ap.add_argument("--lang", default="en", help="two-letter language code to use as the default language")
        add_boolean_option(ap, "multilingual", "separate model parameters per language (passage.attrib['lang'])")

        group = ap.add_argument_group(title="Sanity checks")
        add_boolean_option(group, "check-loops", "check for parser state loop")
        add_boolean_option(group, "verify", "check for oracle reproducing original passage")
        add_param_arguments(ap)

        group = ap.add_argument_group(title="DyNet parameters")
        group.add_argument("--dynet-mem", help="memory for dynet")
        group.add_argument("--dynet-weight-decay", type=float, default=1e-5, help="weight decay for parameters")
        add_boolean_option(group, "dynet-apply-weight-decay-on-load", "workaround for clab/dynet#1206", default=False)
        add_boolean_option(group, "dynet-gpu", "GPU for training")
        group.add_argument("--dynet-gpus", type=int, default=1, help="how many GPUs you want to use")
        add_boolean_option(group, "dynet-autobatch", "auto-batching of training examples")
        DYNET_ARG_NAMES.update(get_group_arg_names(group))

        ap.add_argument("-H", "--hyperparams", type=HyperparamsInitializer.action, nargs="*",
                        help="shared hyperparameters or hyperparameters for specific formats, "
                             'e.g., "shared --lstm-layer-dim=100 --lstm-layers=1" "ucca --word-dim=300"',
                        default=[HyperparamsInitializer.action("shared --lstm-layers 2")])
        ap.add_argument("--copy-shared", nargs="*", choices=FORMATS, help="formats whose parameters shall be "
                                                                          "copied from loaded shared parameters")
        self.args = FallbackNamespace(ap.parse_args(args if args else None))

        if self.args.config:
            print("Loading configuration from '%s'." % self.args.config)

        if self.args.passages and self.args.write:
            os.makedirs(self.args.outdir, exist_ok=True)

        if self.args.models:
            if not self.args.log:
                self.args.log = self.args.models[0] + ".log"
            if self.args.dev and not self.args.devscores:
                self.args.devscores = self.args.models[0] + ".dev.csv"
            if self.args.passages and not self.args.testscores:
                self.args.testscores = self.args.models[0] + ".test.csv"
        elif not self.args.log:
            self.args.log = "parse.log"
        self.sub_configs = []  # Copies to be stored in Models so that they do not interfere with each other
        self._logger = self.format = self.hyperparams = self.iteration_hyperparams = None
        self._vocab = {}
        self.original_values = {}
        self.random = np.random
        self.update()

    def create_original_values(self, args=None):
        return {attr: getattr(self.args, attr) if args is None else args[attr]
                for attr in RESTORED_ARGS if args is None or attr in args}

    def set_format(self, f=None, update=False):
        if f in (None, "text") and not self.format:  # In update or parsing UCCA (with no extra["format"]) or plain text
            f = "ucca"  # Default output format is UCCA
        if update or self.format != f:
            if f not in (None, "text"):
                self.format = f
            self.update_by_hyperparams()
        for config in self.sub_configs:
            config.set_format(f=f, update=update)

    def is_unlabeled(self, f=None):
        # If just -u or --unlabeled is given then its value is [], and we want to treat that as "all formats"
        # If not given at all it is None, and we want to treat that as "no format"
        return self.args.unlabeled == [] or (f or self.format) in (self.args.unlabeled or ())

    def max_actions_unlabeled(self):
        return 6 + (  # Shift Node Reduce LeftEdge RightEdge Finish
            3 if self.args.remote else 0) + (  # RemoteNode LeftRemote RightRemote
            1 if self.args.swap == REGULAR else (self.args.max_swap if self.args.swap == COMPOUND else 0)) + (  # Swap
            1 if self.args.implicit else 0) + (  # Implicit
            2 if self.args.node_labels and not self.args.use_gold_node_labels else 0)  # Label x 2

    def set_dynet_arguments(self):
        self.random.seed(self.args.seed)
        kwargs = dict(random_seed=self.args.seed)
        if self.args.dynet_mem:
            kwargs.update(mem=self.args.dynet_mem)
        if self.args.dynet_weight_decay:
            kwargs.update(weight_decay=self.args.dynet_weight_decay)
        if self.args.dynet_gpus and self.args.dynet_gpus != 1:
            kwargs.update(requested_gpus=self.args.dynet_gpus)
        if self.args.dynet_autobatch:
            kwargs.update(autobatch=True)
        dynet_config.set(**kwargs)
        if self.args.dynet_gpu:
            dynet_config.set_gpu()

    def update(self, params=None):
        if params:
            for name, value in params.items():
                setattr(self.args, name, value)
        self.original_values.update(self.create_original_values(params))
        self.hyperparams = self.create_hyperparams()
        for f, num in EDGE_LABELS_NUM.items():
            self.hyperparams.specific[f].max_edge_labels = num
        amr_hyperparams = self.hyperparams.specific["amr"]
        for k, v in dict(node_label_dim=20, max_node_labels=1000, node_category_dim=5, max_node_categories=25).items():
            if k not in amr_hyperparams and not getattr(amr_hyperparams, k, None):
                setattr(amr_hyperparams, k, v)
        self.set_format(update=True)
        self.set_dynet_arguments()

    def create_hyperparams(self):
        return Hyperparams(parent=self.args, **{h.name: h.args for h in self.args.hyperparams or ()})

    def update_hyperparams(self, **kwargs):
        self.update(dict(hyperparams=[HyperparamsInitializer(k, **v) for k, v in kwargs.items()]))

    def update_iteration(self, iteration, print_message=True):
        if iteration.hyperparams:
            if print_message:
                print("Updating: %s" % iteration.hyperparams)
            self.iteration_hyperparams = iteration.hyperparams.args
            self.update_by_hyperparams()
            for config in self.sub_configs:
                config.update_iteration(iteration, print_message=False)

    def update_by_hyperparams(self):
        format_values = dict(self.original_values)
        for hyperparams in (self.iteration_hyperparams, self.hyperparams.specific[self.format]):
            if hyperparams:
                format_values.update({k: v for k, v in hyperparams.items() if not k.startswith("_")})
        for attr, value in sorted(format_values.items()):
            self.print("Setting %s=%s" % (attr, value))
            setattr(self.args, attr, value)
        if self.format != "amr":
            self.args.node_labels = False
            self.args.node_label_dim = self.args.max_node_labels = \
                self.args.node_category_dim = self.args.max_node_categories = 0
        if self.is_unlabeled():
            self.args.max_edge_labels = self.args.edge_label_dim = 0
            self.args.max_action_labels = self.max_actions_unlabeled()
        else:
            self.args.max_action_labels = max(self.args.max_action_labels, 6 * self.args.max_edge_labels)

    @property
    def line_end(self):
        return "\n" if self.args.verbose > 2 else " "  # show all in one line unless verbose

    @property
    def passage_word(self):
        return "sentence" if self.args.sentences else "paragraph" if self.args.paragraphs else "passage"

    @property
    def passages_word(self):
        return " %ss" % self.passage_word

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

    def vocab(self, filename=None, lang=None):
        if filename is None:
            args = self.hyperparams.specific[lang] if lang else self.args
            filename = args.vocab
        if not filename:
            return None
        vocab = self._vocab.get(filename)
        if vocab:
            return vocab
        vocab = load_enum(filename)
        self._vocab[filename] = vocab
        return vocab

    def print(self, message, level=3):
        if self.args.verbose >= level:
            try:
                print(message() if hasattr(message, "__call__") else message, flush=True)
            except UnicodeEncodeError:
                pass

    def save(self, filename):
        out_file = filename + ".yml"
        print("Saving configuration to '%s'." % out_file)
        with open(out_file, "w") as f:
            name = None
            values = []
            for arg in shlex.split(str(self), "--") + ["--"]:
                if arg.startswith("--"):
                    if name and name not in ("train", "dev"):
                        if len(values) > 1:
                            values[0] = "[" + values[0]
                            values[-1] += "]"
                        elif name.startswith("no-"):
                            name = name[3:]
                            values = ["false"]
                        print("%s: %s" % (name, ", ".join(values) or "true"), file=f)
                    name = arg[2:]
                    values = []
                else:
                    values.append(arg)

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        ret.arg_parser = self.arg_parser
        ret.args = deepcopy(self.args)
        ret.original_values = deepcopy(self.original_values)
        ret.hyperparams = deepcopy(self.hyperparams)
        ret.iteration_hyperparams = deepcopy(self.iteration_hyperparams)
        ret.format = self.format
        ret.random = self.random
        ret._logger = self._logger
        ret._vocab = dict(self._vocab)
        ret.sub_configs = []
        self.sub_configs.append(ret)
        return ret

    def args_str(self, args):
        return ["--" + ("no-" if v is False else "") + k.replace("_", "-") +
                ("" if v is False or v is True else
                 (" " + str(" ".join(map(str, v)) if hasattr(v, "__iter__") and not isinstance(v, str) else v)))
                for (k, v) in sorted(args.items()) if
                v not in (None, (), "", self.arg_parser.get_default(k))
                and not k.startswith("_")
                and (args.node_labels or ("node_label" not in k and "node_categor" not in k))
                and (args.swap or "swap_" not in k)
                and (args.swap == COMPOUND or k != "max_swap")
                and (not args.require_connected or k != "orphan_label")
                and (args.classifier == SPARSE or k not in SPARSE_ARG_NAMES)
                and (args.classifier in NN_CLASSIFIERS or k not in NN_ARG_NAMES | DYNET_ARG_NAMES)
                and k != "passages"]

    def __str__(self):
        self.args.hyperparams = [HyperparamsInitializer(name, **args.vars()) for name, args in self.hyperparams.items()]
        return " ".join(list(self.args.passages) + self.args_str(self.args))
