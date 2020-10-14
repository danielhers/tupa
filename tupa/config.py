import shlex
import sys
from copy import copy

import dynet_config
import numpy as np
from configargparse import ArgParser, Namespace, ArgumentDefaultsHelpFormatter, SUPPRESS, FileType, Action
from logbook import Logger, FileHandler, StderrHandler

from .classifiers.nn.constants import *
from .model_util import load_enum

# Classifiers
BIRNN = "bilstm"
NOOP = "noop"
CLASSIFIERS = (BIRNN, NOOP)

FEATURE_PROPERTIES = "wmtudhenpqxyANEPC"

# Swap types
REGULAR = "regular"
COMPOUND = "compound"

# Required number of edge labels per framework
NODE_LABELS_NUM = {"amr": 1000, "dm": 1000, "psd": 1000, "eds": 1000, "ucca": 0, "ptg": 1000}
NODE_PROPERTY_NUM = {"amr": 1000, "dm": 510, "psd": 1000, "eds": 1000, "ucca": 0, "ptg": 1000}
EDGE_LABELS_NUM = {"amr": 141, "dm": 59, "psd": 90, "eds": 10, "ucca": 15, "ptg": 150}
EDGE_ATTRIBUTE_NUM = {"amr": 0, "dm": 0, "psd": 0, "eds": 0, "ucca": 2}
NN_ARG_NAMES = set()
DYNET_ARG_NAMES = set()
RESTORED_ARGS = set()

SEPARATOR = "."


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance

    def reload(cls):
        cls.instance = None


class VAction(Action):
    def __call__(self, parser, args, values, option_string=None):
        if values is None:
            values = "1"
        try:
            values = int(values)
        except ValueError:
            values = values.count("v") + 1
        setattr(args, self.dest, values)


def add_verbose_arg(argparser, **kwargs):
    return argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, **kwargs)


def get_group_arg_names(group):
    return [a.dest for a in group._group_actions]


def add_boolean_option(argparser, name, description, default=False, short=None, short_no=None):
    group = argparser.add_mutually_exclusive_group()
    options = [] if short is None else ["-" + short]
    options.append("--" + name)
    group.add_argument(*options, action="store_true", default=default, help="include " + description)
    no_options = [] if short_no is None else ["-" + short_no]
    no_options.append("--no-" + name)
    group.add_argument(*no_options, action="store_false", dest=name.replace("-", "_"), default=default,
                       help="exclude " + description)
    return group


def add_param_arguments(ap=None, arg_default=None):  # arguments with possible framework-specific parameter values

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
    add(group, "--max-node-labels", type=int, default=1000, help="max number of node labels to allow")
    add(group, "--min-node-label-count", type=int, default=2, help="min number of occurrences for a label")

    group = ap.add_argument_group(title="Node properties")
    add(group, "--max-node-properties", type=int, default=100, help="max number of node property values to allow")
    add(group, "--min-node-property-count", type=int, default=2, help="min number of occurrences for a property")
    add(group, "--max-properties-per-node", type=int, default=4, help="max number of properties to allow on one node")

    group = ap.add_argument_group(title="Edge attributes")
    add(group, "--max-edge-attributes", type=int, default=2, help="max number of edge attribute values to allow")
    add(group, "--max-attributes-per-edge", type=int, default=1, help="max number of attributes to allow on one edge")

    group = ap.add_argument_group(title="Structural constraints")
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
    add(group, "--max-training-per-framework", type=int,
        help="max number of training graphs per framework per iteration")
    add_boolean(group, "missing-node-features", "allow node features to be missing if not available", default=True)
    add(group, "--omit-features", help="string of feature properties to omit, out of " + FEATURE_PROPERTIES)
    add_boolean(group, "curriculum", "sort training graphs by action prediction accuracy in previous epoch")

    group = ap.add_argument_group(title="Neural network parameters")
    add(group, "--word-dim-external", type=int, default=0, help="dimension for external word embeddings")
    add(group, "--word-vectors", help="file to load external word embeddings from (default: GloVe)")
    add(group, "--vocab", help="file mapping integer ID to word form (to avoid loading spaCy), or '-' to use word form")
    add_boolean(group, "update-word-vectors", "external word vectors in training parameters", default=True)
    add(group, "--word-dim", type=int, default=0, help="dimension for learned word embeddings")
    add(group, "--lemma-dim", type=int, default=200, help="dimension for lemma embeddings")
    add(group, "--tag-dim", type=int, default=20, help="dimension for fine POS tag embeddings")
    add(group, "--pos-dim", type=int, default=20, help="dimension for coarse/universal POS tag embeddings")
    add(group, "--dep-dim", type=int, default=10, help="dimension for dependency relation embeddings")
    add(group, "--edge-label-dim", type=int, default=20, help="dimension for edge label embeddings")
    add(group, "--edge-attribute-dim", type=int, default=1, help="dimension for edge attribute embeddings")
    add(group, "--node-label-dim", type=int, default=20, help="dimension for node label embeddings")
    add(group, "--node-property-dim", type=int, default=20, help="dimension for node property embeddings")
    add(group, "--punct-dim", type=int, default=1, help="dimension for separator punctuation embeddings")
    add(group, "--action-dim", type=int, default=3, help="dimension for input action type embeddings")
    add(group, "--output-dim", type=int, default=50, help="dimension for output action embeddings")
    add(group, "--layer-dim", type=int, default=50, help="dimension for hidden layers")
    add(group, "--layers", type=int, default=2, help="number of hidden layers")
    add(group, "--lstm-layer-dim", type=int, default=300, help="dimension for LSTM hidden layers")
    add(group, "--lstm-layers", type=int, default=2, help="number of LSTM hidden layers")
    add(group, "--embedding-layer-dim", type=int, default=300, help="dimension for layers before LSTM")
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
    add(group, "--word-dropout", type=float, default=0.2, help="word dropout parameter")
    add(group, "--word-dropout-external", type=float, default=0, help="word dropout for word vectors")
    add(group, "--lemma-dropout", type=float, default=0.2, help="lemma dropout parameter")
    add(group, "--tag-dropout", type=float, default=0.2, help="fine POS tag dropout parameter")
    add(group, "--pos-dropout", type=float, default=0.2, help="coarse POS tag dropout parameter")
    add(group, "--dep-dropout", type=float, default=0.5, help="dependency label dropout parameter")
    add(group, "--node-label-dropout", type=float, default=0.2, help="node label dropout parameter")
    add(group, "--node-property-dropout", type=float, default=0.2, help="node property dropout parameter")
    add(group, "--edge-attribute-dropout", type=float, default=0.2, help="edge attribute dropout parameter")
    add(group, "--node-dropout", type=float, default=0.1, help="probability to drop features for a whole node")
    add(group, "--dropout", type=float, default=0.4, help="dropout parameter between layers")
    add(group, "--max-length", type=int, default=80, help="maximum length of input sentence")
    add(group, "--rnn", choices=["None"] + list(RNNS), default=DEFAULT_RNN, help="type of recurrent neural network")
    add(group, "--gated", type=int, nargs="?", default=2, help="gated input to BiRNN and MLP")
    add(group, "--use-bert", action="store_true", help="whether to use bert embeddings")
    add(group, "--bert-model", choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                        "bert-large-cased", "bert-base-multilingual-cased"],
        default="bert-large-cased")
    add(group, "--bert-layers", type=int, nargs='+', default=[-1, -2, -3, -4])
    add(group, "--bert-layers-pooling", choices=["weighed", "sum", "concat"], default="weighed")
    add(group, "--bert-token-align-by", choices=["first", "sum", "mean"], default="sum")
    add(group, "--bert-multilingual", choices=[0], type=int)
    add(group, "--use-default-word-embeddings", action="store_true", help="whether to use external word vectors")
    add(group, "--bert-dropout", type=float, default=0, choices=np.linspace(0, 0.9, num=10))
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
        self.arg_parser = ap = ArgParser(description="Transition-based meaning representation parser.",
                                         formatter_class=ArgumentDefaultsHelpFormatter)
        ap.add_argument("input", nargs="?", type=FileType("r", encoding="utf-8"), default=sys.stdin,
                        help="file with one MRP per line")
        ap.add_argument("output", nargs="?", type=FileType("w", encoding="utf-8"), default=sys.stdout,
                        help="output file to create")
        ap.add_argument("--version", action="version", version="")
        ap.add_argument("-C", "--config", is_config_file=True, help="configuration file to get arguments from")
        ap.add_argument("--conllu", type=FileType("r", encoding="utf-8"),
                        help="file with one MRP per line, to get conllu features from")
        ap.add_argument("--alignment", type=FileType("r", encoding="utf-8"), help="file to get AMR alignments from")
        ap.add_argument("-m", "--model", help="model file basename to load/save (default: <framework>_<model_type>")
        ap.add_argument("-c", "--classifier", choices=CLASSIFIERS, default=BIRNN, help="model type")
        add_boolean_option(ap, "evaluate", "evaluation of parsed graphs", short="e")
        add_verbose_arg(ap, help="detailed parse output")
        ap.add_argument("--timeout", type=float, help="max number of seconds to wait for a single graph")
        ap.add_argument("--cores", type=int, default=1, help="number of CPU cores to use for running the evaluator")

        group = ap.add_argument_group(title="Training parameters")
        group.add_argument("-t", "--train", action="store_true", help="train a model on the input")
        group.add_argument("-d", "--dev", type=FileType("r", encoding="utf-8"),
                           help="graph files/directories to tune on")
        group.add_argument("-I", "--iterations", nargs="+", type=Iterations,
                           default=(Iterations(50), Iterations("100 --optimizer=" + EXTRA_TRAINER)),
                           help="number of training iterations along with optional hyperparameters per part")
        group.add_argument("--folds", type=int, choices=(3, 5, 10), help="#folds for cross validation")
        group.add_argument("--seed", type=int, default=1, help="random number generator seed")
        group.add_argument("--save-every", type=int, help="every this many graphs, evaluate on dev and save model")
        add_boolean_option(group, "eval-test", "evaluate on test whenever evaluating on dev, but keep results hidden")

        group = ap.add_argument_group(title="Output files")
        group.add_argument("-p", "--prefix", default="", help="output filename prefix")
        add_boolean_option(group, "write", "writing parsed output to files", default=True, short_no="W")
        group.add_argument("-j", "--join",
                           help="if output framework is textual, write all to one file with this basename")
        group.add_argument("-l", "--log", help="output log file (default: model filename + .log)")
        group.add_argument("--devscores", help="output CSV file for dev scores (default: model filename + .dev.csv)")
        group.add_argument("--testscores", help="output CSV file for test scores (default: model filename + .test.csv)")
        group.add_argument("--diagnostics", help="output CSV file for diagnostics info (default: model filename + "
                                                 ".diagnostics.csv)")
        group.add_argument("--action-stats", help="output CSV file for action statistics")

        group = ap.add_argument_group(title="Sanity checks")
        add_boolean_option(group, "check-loops", "check for parser state loop")
        add_boolean_option(group, "validate-oracle", "require oracle output to respect constraints", default=True)
        add_param_arguments(ap)

        group = ap.add_argument_group(title="DyNet parameters")
        group.add_argument("--dynet-mem", help="memory for dynet")
        group.add_argument("--dynet-weight-decay", type=float, default=1e-5, help="weight decay for parameters")
        add_boolean_option(group, "dynet-apply-weight-decay-on-load", "workaround for clab/dynet#1206", default=False)
        add_boolean_option(group, "dynet-gpu", "GPU for training")
        add_boolean_option(group, "pytorch-gpu", "GPU for BERT")
        group.add_argument("--dynet-gpus", type=int, default=1, help="how many GPUs you want to use")
        add_boolean_option(group, "dynet-autobatch", "auto-batching of training examples")
        add_boolean_option(group, "dynet-check-validity", "check validity of expressions immediately")
        DYNET_ARG_NAMES.update(get_group_arg_names(group))

        ap.add_argument("-H", "--hyperparams", type=HyperparamsInitializer.action, nargs="*",
                        help="shared hyperparameters or hyperparameters for specific frameworks, "
                             'e.g., "shared --lstm-layer-dim=100 --lstm-layers=1" "ucca --word-dim=300"',
                        default=[HyperparamsInitializer.action("shared --lstm-layers 2")])
        self.args = FallbackNamespace(ap.parse_args(args if args else None))

        if self.args.config:
            print("Loading configuration from '%s'." % self.args.config, file=sys.stderr)

        if self.args.model:
            if not self.args.log:
                self.args.log = self.args.model + ".log"
            if self.args.dev and not self.args.devscores:
                self.args.devscores = self.args.model + ".dev.csv"
            if self.args.input and not self.args.testscores:
                self.args.testscores = self.args.model + ".test.csv"
            if self.args.input and not self.args.diagnostics:
                self.args.diagnostics = self.args.model + ".diagnostics.csv"
            if self.args.diagnostics:
                with open(self.args.diagnostics, "a") as f:
                    print("id", "framework", "tokens", "actions", sep=",", file=f)
        elif not self.args.log:
            self.args.log = "parse.log"
        self.sub_configs = []  # Copies to be stored in Models so that they do not interfere with each other
        self._logger = self.framework = self.hyperparams = self.iteration_hyperparams = None
        self._vocab = {}
        self.original_values = {}
        self.random = np.random
        self.update()

    def create_original_values(self, args=None):
        return {attr: getattr(self.args, attr) if args is None else args[attr]
                for attr in RESTORED_ARGS if args is None or attr in args}

    def set_framework(self, framework=None, update=False, recursive=True):
        if update or self.framework != framework:
            if framework is not None:
                self.framework = framework
            self.update_by_hyperparams()
        if recursive:
            for config in self.descendants():
                config.set_framework(framework=framework, update=update, recursive=False)

    def descendants(self):
        ret = []
        configs = [self]
        while configs:
            c = configs.pop(0)
            ret += c.sub_configs
            configs += c.sub_configs
        return ret

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
        for framework, num in NODE_LABELS_NUM.items():
            self.hyperparams.specific[framework].max_node_labels = num
        for framework, num in EDGE_LABELS_NUM.items():
            self.hyperparams.specific[framework].max_edge_labels = num
        for framework, num in NODE_PROPERTY_NUM.items():
            self.hyperparams.specific[framework].max_node_properties = num
        for framework, num in EDGE_ATTRIBUTE_NUM.items():
            self.hyperparams.specific[framework].max_edge_attributes = num
        self.set_framework(update=True)
        self.set_dynet_arguments()

    def create_hyperparams(self):
        return Hyperparams(parent=self.args, **{h.name: h.args for h in self.args.hyperparams or ()})

    def update_hyperparams(self, **kwargs):
        self.update(dict(hyperparams=[HyperparamsInitializer(k, **v) for k, v in kwargs.items()]))

    def update_iteration(self, iteration, print_message=True, recursive=True):
        if iteration.hyperparams:
            if print_message:
                print("Updating: %s" % iteration.hyperparams, file=sys.stderr)
            self.iteration_hyperparams = iteration.hyperparams.args
            self.update_by_hyperparams()
            if recursive:
                for config in self.descendants():
                    config.update_iteration(iteration, print_message=False, recursive=False)

    def update_by_hyperparams(self):
        format_values = dict(self.original_values)
        for hyperparams in (self.iteration_hyperparams, self.hyperparams.specific[self.framework]):
            if hyperparams:
                format_values.update({k: v for k, v in hyperparams.items() if not k.startswith("_")})
        for attr, value in sorted(format_values.items()):
            self.print("Setting %s=%s" % (attr, value))
            setattr(self.args, attr, value)
        self.args.max_action_labels = max(self.args.max_action_labels, 4 * self.args.max_edge_labels)

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

    def vocab(self, filename=None):
        if filename is None:
            filename = self.args.vocab
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
                print(message() if hasattr(message, "__call__") else message, flush=True, file=sys.stderr)
            except UnicodeEncodeError:
                pass

    def save(self, filename):
        if filename is not None:
            out_file = filename + ".yml"
            print("Saving configuration to '%s'." % out_file, file=sys.stderr)
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
        ret.args = copy(self.args)
        ret.original_values = copy(self.original_values)
        ret.hyperparams = copy(self.hyperparams)
        ret.iteration_hyperparams = copy(self.iteration_hyperparams)
        ret.framework = self.framework
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
                and (args.swap or "swap_" not in k)
                and (args.swap == COMPOUND or k != "max_swap")
                and (not args.require_connected or k != "orphan_label")
                and (args.classifier == BIRNN or k not in NN_ARG_NAMES | DYNET_ARG_NAMES)
                and k not in ("input", "output")]

    def __str__(self):
        self.args.hyperparams = [HyperparamsInitializer(name, **args.vars()) for name, args in self.hyperparams.items()]
        return " ".join([self.args.input.name, self.args.output.name] + self.args_str(self.args))


def requires_node_labels(framework):
    return framework not in ("ucca", "drg", "ptg")


def requires_node_properties(framework):
    return framework not in ("ucca", "drg")


def requires_edge_attributes(framework):
    return framework == "ucca"


def requires_anchors(framework):
    return framework not in ("amr", "drg")


def requires_tops(framework):
    return framework in ("ucca", "amr", "drg", "ptg")
