from collections import OrderedDict

from .action import Actions
from .classifiers.classifier import Classifier
from .config import Config, NOOP, requires_node_labels, requires_node_properties, requires_edge_attributes
from .features.feature_params import FeatureParameters
from .model_util import UnknownDict, AutoIncrementDict, remove_backup


class ParameterDefinition:
    def __init__(self, args, name, attr_to_arg, attr_to_val=None):
        self.args = args
        self.name = name
        self.attr_to_arg = attr_to_arg
        self.attr_to_val = attr_to_val or {}

    @property
    def dim_arg(self):
        return self.attr_to_arg["dim"]

    @property
    def size_arg(self):
        return self.attr_to_arg["size"]

    @property
    def enabled(self):
        return bool(getattr(self.args, self.dim_arg) and getattr(self.args, self.size_arg))
    
    @enabled.setter
    def enabled(self, value):
        if value:
            raise ValueError("Can only disable parameter configuration by setting 'enabled' to False")
        setattr(self.args, self.dim_arg, 0)

    def create_from_config(self):
        kwargs = dict(self.attr_to_val)
        kwargs.update({k: getattr(self.get_args(), v) for k, v in self.attr_to_arg.items()})
        return FeatureParameters(self.name, **kwargs)

    def load_to_config(self, params):
        param = params.get(self.key())
        self.get_args().update({self.dim_arg: 0, self.size_arg: 0} if param is None else
                               {v: getattr(param, k) for k, v in self.attr_to_arg.items()})

    def get_args(self):
        return self.args

    def key(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s(%s, %s)" % (type(self).__name__, self.name, ", ".join(
            "%s=%s" % i for i in list(self.attr_to_arg.items()) + list(self.attr_to_val.items())))


# Output keys that are also feature parameter keys
NODE_LABEL_KEY = "n"
NODE_PROPERTY_KEY = "N"
EDGE_ATTRIBUTE_KEY = "E"
SHARED_OUTPUT_KEYS = (NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY)


SHARED_OUTPUT_PARAM_DEFS = [
    (NODE_LABEL_KEY, dict(dim="node_label_dim",    size="max_node_labels",    dropout="node_label_dropout",
                          min_count="min_node_label_count")),
    (NODE_PROPERTY_KEY, dict(dim="node_property_dim", size="max_node_properties", dropout="node_property_dropout",
                             min_count="min_node_property_count")),
    (EDGE_ATTRIBUTE_KEY, dict(dim="edge_attribute_dim", size="max_edge_attributes", dropout="edge_attribute_dropout")),
]
PARAM_DEFS = [
    ("W",            dict(dim="word_dim_external", size="max_words_external", dropout="word_dropout_external",
                          updated="update_word_vectors", filename="word_vectors", vocab="vocab"), dict(
                                                                                 copy_from="w")),
    ("w",            dict(dim="word_dim",       size="max_words",  dropout="word_dropout")),
    ("m",            dict(dim="lemma_dim",      size="max_lemmas", dropout="lemma_dropout")),
    ("t",            dict(dim="tag_dim",        size="max_tags",   dropout="tag_dropout")),
    ("u",            dict(dim="pos_dim",        size="max_pos",    dropout="pos_dropout")),
    ("d",            dict(dim="dep_dim",        size="max_deps",   dropout="dep_dropout")),
    ("e",            dict(dim="edge_label_dim", size="max_edge_labels")),
    ("p",            dict(dim="punct_dim",      size="max_puncts")),
    ("A",            dict(dim="action_dim",     size="max_action_types")),
]


class Model:
    def __init__(self, filename=None, config=None, *args, **kwargs):
        self.config = config or Config().copy()
        self.filename = filename
        self.feature_extractor = self.classifier = None
        self.feature_params = OrderedDict()
        self.is_finalized = False
        if args or kwargs:
            self.restore(*args, **kwargs)

    def param_defs(self, args=None, only=None):
        return [ParameterDefinition(args or self.config.args, key, *values) for key, *values in
                SHARED_OUTPUT_PARAM_DEFS + ([] if only is not None else PARAM_DEFS)
                if only is None or key == only]

    def init_model(self, framework=None, init_params=True):
        outputs = self.classifier.labels if self.classifier else OrderedDict()
        if init_params:  # Actually use the config state to initialize the features and hyperparameters, otherwise empty
            for param_def in self.param_defs():  # FIXME save parameters separately per framework
                key = param_def.key()
                param = self.feature_params.get(key)
                enabled = param_def.enabled
                if param:
                    param.enabled = enabled
                elif enabled:
                    self.feature_params[key] = param_def.create_from_config()
                    self.init_param(key)
            if framework is not None and framework not in outputs:
                outputs[framework] = self.init_actions()  # Uses config to determine actions
            # Update self.feature_params:
            self.init_node_labels(framework, outputs)
            self.init_node_properties(framework, outputs)
            self.init_edge_attributes(framework, outputs)
        if self.classifier:  # Already initialized
            pass
        elif self.config.args.classifier == NOOP:
            from .features.empty_features import EmptyFeatureExtractor
            from .classifiers.noop import NoOp
            self.feature_extractor = EmptyFeatureExtractor()
            self.classifier = NoOp(self.config, outputs)
        else:
            from .features.dense_features import DenseFeatureExtractor
            from .classifiers.nn.neural_network import NeuralNetwork
            self.feature_extractor = DenseFeatureExtractor(self.feature_params,
                                                           node_dropout=self.config.args.node_dropout,
                                                           omit_features=self.config.args.omit_features)
            self.classifier = NeuralNetwork(self.config, outputs)
        self._update_input_params()

    def output_values(self, framework, key=None):
        return self.classifier.labels[self.key(framework, key)]

    @staticmethod
    def key(framework, key=None):
        return "_".join(filter(None, (framework, key)))

    def init_actions(self):
        return Actions(size=self.config.args.max_action_labels)

    def init_param(self, key):
        feature_param = self.feature_params.get(key)
        if feature_param is None:
            feature_param = self.param_defs(only=key)[0].create_from_config()
            self.feature_params[key] = feature_param
        if self.feature_extractor:
            self.feature_extractor.init_param(key)
        feature_param.init_data()
        return feature_param

    def init_outputs(self, framework, key, outputs):
        axis = self.key(framework, key)
        if axis not in outputs:
            outputs[axis] = self.init_param(key).data

    def init_node_labels(self, framework, outputs):
        if requires_node_labels(framework):
            self.init_outputs(framework, NODE_LABEL_KEY, outputs)

    def init_node_properties(self, framework, outputs):
        if requires_node_properties(framework):
            self.init_outputs(framework, NODE_PROPERTY_KEY, outputs)

    def init_edge_attributes(self, framework, outputs):
        if requires_edge_attributes(framework):
            self.init_outputs(framework, EDGE_ATTRIBUTE_KEY, outputs)

    def score(self, state, framework, key):
        features = self.feature_extractor.extract_features(state)
        return self.classifier.score(features, axis=self.key(framework, key)), features  # scores is a NumPy array

    def init_features(self, framework, state, train):
        self.init_model(framework)
        axes = [framework]
        if requires_node_labels(state.framework):
            axes.append(self.key(framework, NODE_LABEL_KEY))
        if requires_node_properties(state.framework):
            axes.append(self.key(framework, NODE_PROPERTY_KEY))
        if requires_edge_attributes(state.framework):
            axes.append(self.key(framework, EDGE_ATTRIBUTE_KEY))
        tokens = [terminal.text for terminal in state.terminals]
        lang = getattr(state.input_graph, "lang", "en")
        self.classifier.init_features(self.feature_extractor.init_features(state), axes, train, tokens, lang)

    def finalize(self, finished_epoch):
        """
        Copy model, finalizing features (new values will not be added during subsequent use) and classifier (update it)
        :param finished_epoch: whether this is the end of an epoch (or just intermediate checkpoint), for bookkeeping
        :return: a copy of this model with a new feature extractor and classifier (actually classifier may be the same)
        """
        self.config.print("Finalizing model", level=1)
        return Model(None, config=self.config.copy(), model=self, is_finalized=True,
                     feature_extractor=self.feature_extractor.finalize(),
                     classifier=self.classifier.finalize(finished_epoch=finished_epoch))

    def save(self, save_init=False):
        """
        Save feature and classifier parameters to files
        """
        if self.filename is not None:
            try:
                self.feature_extractor.save(self.filename, save_init=save_init)
                skip_labels = []
                for key in SHARED_OUTPUT_KEYS:
                    feature_param = self.feature_extractor.params.get(key)
                    skip_labels.append((key,) if feature_param and feature_param.size else ())
                self.classifier.save(self.filename, skip_labels=skip_labels,
                                     omit_features=self.config.args.omit_features)
                remove_backup(self.filename)
            except Exception as e:
                raise IOError("Failed saving model to '%s'" % self.filename) from e

    def load(self, is_finalized=True):
        """
        Load the feature and classifier parameters from files
        :param is_finalized: whether loaded model should be finalized, or allow feature values to be added subsequently
        """
        if self.filename is not None:
            try:
                self.config.args.classifier = Classifier.get_property(self.filename, "type")
                self.config.args.omit_features = Classifier.get_property(self.filename, "omit_features")
                self.init_model(init_params=False)
                self.feature_extractor.load(self.filename, order=[p.name for p in self.param_defs()])
                if not is_finalized:
                    self.feature_extractor.unfinalize()
                self._update_input_params()  # Must be before classifier.load() because it uses them to init the model
                self.classifier.load(self.filename)
                self.is_finalized = is_finalized
                self.load_labels()
                for param_def in self.param_defs(self.config):
                    param_def.load_to_config(self.feature_extractor.params)
                self.config.print("\n".join("%s: %s" % i for i in self.feature_params.items()), level=1)
            except FileNotFoundError:
                self.feature_extractor = self.classifier = None
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e

    def restore(self, model, feature_extractor=None, classifier=None, is_finalized=None):
        """
        Set all attributes to a reference to existing model, except labels, which will be copied.
        :param model: Model to restore
        :param feature_extractor: optional FeatureExtractor to restore instead of model's
        :param classifier: optional Classifier to restore instead of model's
        :param is_finalized: whether the restored model is finalized
        """
        if is_finalized is None:
            is_finalized = model.is_finalized
        self.config.print("Restoring %sfinalized model" % ("" if is_finalized else "non-"), level=1)
        self.filename = model.filename
        self.feature_extractor = feature_extractor or model.feature_extractor
        self.classifier = classifier or model.classifier
        self.is_finalized = is_finalized
        self._update_input_params()
        self.classifier.labels_t = OrderedDict((a, l.save()) for a, l in self.classifier.labels.items())
        self.load_labels()

    def load_labels(self):
        """
        Copy classifier's labels to create new Actions/Labels objects
        Restoring from a model that was just loaded from file, or called by restore()
        """
        for axis, all_size in self.classifier.labels_t.items():  # all_size is a pair of (label list, size limit)
            framework, _, key = axis.partition("_")
            if key in SHARED_OUTPUT_KEYS:  # These are labels/properties/attributes rather than action labels
                feature_param = self.feature_extractor.params.get(key)
                if feature_param and feature_param.size:  # Also used for features, so share the dict
                    del all_size
                    labels = feature_param.data
                else:  # Not used as a feature, just get labels
                    labels = UnknownDict() if self.is_finalized else AutoIncrementDict()
                    labels.load(all_size)
            else:  # Action labels for framework determined by axis
                labels = Actions(*all_size)
            self.classifier.labels[axis] = labels

    def _update_input_params(self):
        self.feature_params = self.classifier.input_params = self.feature_extractor.params

    def all_params(self):
        d = OrderedDict()
        d["features"] = self.feature_extractor.all_features()
        d.update(("input_" + k, p.data.all) for k, p in self.feature_extractor.params.items() if p.data)
        d.update(self.classifier.all_params())
        return d
