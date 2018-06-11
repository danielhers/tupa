from collections import OrderedDict

from enum import Enum
from ucca import textutil

from .action import Actions
from .classifiers.classifier import Classifier
from .config import Config, SEPARATOR, SPARSE, MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN, NOOP
from .features.feature_params import FeatureParameters
from .model_util import UnknownDict, AutoIncrementDict, remove_backup, save_json, load_json


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
        return bool(getattr(self.args, self.dim_arg))
    
    @enabled.setter
    def enabled(self, value):
        if value:
            raise ValueError("Can only disable parameter configuration by setting 'enabled' to False")
        setattr(self.args, self.dim_arg, 0)

    @property
    def lang_specific(self):
        return self.attr_to_val.get("lang_specific")

    def create_from_config(self, lang=None):
        kwargs = dict(self.attr_to_val)
        kwargs.update({k: getattr(self.get_args(lang), v) for k, v in self.attr_to_arg.items()})
        return FeatureParameters(self.name, **kwargs)

    def load_to_config(self, params):
        for lang in list(self.all_langs(params)) or [None]:
            param = params.get(self.key(lang))
            self.get_args(lang).update({self.dim_arg: 0, self.size_arg: 0} if param is None else
                                       {v: getattr(param, k) for k, v in self.attr_to_arg.items()})

    def get_args(self, lang):
        return self.args.hyperparams.specific[lang] if lang else self.args

    def all_langs(self, params):
        for key in params:
            param_name, _, lang = key.partition(SEPARATOR)
            if param_name == self.name and lang:
                yield lang

    def key(self, lang=None):
        return SEPARATOR.join(filter(None, (self.name, lang)))

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s(%s, %s)" % (type(self).__name__, self.name, ", ".join(
            "%s=%s" % i for i in list(self.attr_to_arg.items()) + list(self.attr_to_val.items())))


NODE_LABEL_KEY = "n"


class ClassifierProperty(Enum):
    update_only_on_error = 1
    require_init_features = 2
    trainable_after_saving = 3


CLASSIFIER_PROPERTIES = {
    SPARSE: (ClassifierProperty.update_only_on_error,),
    MLP: (ClassifierProperty.trainable_after_saving,),
    BIRNN: (ClassifierProperty.trainable_after_saving, ClassifierProperty.require_init_features),
    HIGHWAY_RNN: (ClassifierProperty.trainable_after_saving, ClassifierProperty.require_init_features),
    HIERARCHICAL_RNN: (ClassifierProperty.trainable_after_saving, ClassifierProperty.require_init_features),
    NOOP: (ClassifierProperty.trainable_after_saving,),
}

NODE_LABEL_PARAM_DEFS = [
    (NODE_LABEL_KEY, dict(dim="node_label_dim",    size="max_node_labels",    dropout="node_label_dropout",
                          min_count="min_node_label_count"))
]
PARAM_DEFS = [
    ("c",            dict(dim="node_category_dim", size="max_node_categories")),
    ("W",            dict(dim="word_dim_external", size="max_words_external", dropout="word_dropout_external",
                          updated="update_word_vectors", filename="word_vectors", vocab="vocab"), dict(
                                                                                 copy_from="w", lang_specific=True)),
    ("w",            dict(dim="word_dim",       size="max_words",  dropout="word_dropout"),  dict(lang_specific=True)),
    ("m",            dict(dim="lemma_dim",      size="max_lemmas", dropout="lemma_dropout"), dict(lang_specific=True)),
    ("t",            dict(dim="tag_dim",        size="max_tags",   dropout="tag_dropout"),   dict(lang_specific=True)),
    ("u",            dict(dim="pos_dim",        size="max_pos",    dropout="pos_dropout")),
    ("d",            dict(dim="dep_dim",        size="max_deps",   dropout="dep_dropout")),
    ("e",            dict(dim="edge_label_dim", size="max_edge_labels")),
    ("p",            dict(dim="punct_dim",      size="max_puncts")),
    ("A",            dict(dim="action_dim",     size="max_action_types")),
    ("T",            dict(dim="ner_dim",        size="max_ner_types")),
    ("#",            dict(dim="shape_dim",      size="max_shapes"),                          dict(lang_specific=True)),
    ("^",            dict(dim="prefix_dim",     size="max_prefixes"),                        dict(lang_specific=True)),
    ("$",            dict(dim="suffix_dim",     size="max_suffixes"),                        dict(lang_specific=True)),
]


class Model:
    def __init__(self, filename, config=None, *args, **kwargs):
        self.config = config or Config().copy()
        self.filename = filename
        self.feature_extractor = self.classifier = self.axis = self.lang = None
        self.feature_params = OrderedDict()
        self.is_finalized = False
        if args or kwargs:
            self.restore(*args, **kwargs)

    def node_label_param_def(self, args=None):
        return self.param_defs(args, only_node_labels=True)[0]

    def param_defs(self, args=None, only_node_labels=False):
        return [ParameterDefinition(args or self.config.args, n, *k) for n, *k in NODE_LABEL_PARAM_DEFS +
                ([] if only_node_labels else PARAM_DEFS)]

    def init_model(self, axis=None, lang=None, init_params=True):
        self.set_axis(axis, lang)
        labels = self.classifier.labels if self.classifier else OrderedDict()
        if init_params:  # Actually use the config state to initialize the features and hyperparameters, otherwise empty
            for param_def in self.param_defs():  # FIXME save parameters separately per format, not just per language
                for param_lang in (param_def.all_langs(self.feature_params) if self.lang else []) \
                        if param_def.lang_specific and self.config.args.multilingual else [None]:
                    key = param_def.key(param_lang)
                    param = self.feature_params.get(key)
                    enabled = param_def.enabled and (not param_lang or param_lang == self.lang)
                    if param:
                        param.enabled = enabled
                    elif self.is_neural_network and enabled:
                        self.feature_params[key] = param_def.create_from_config(param_lang)
                        self.init_param(key)
            if axis and self.axis not in labels:
                labels[self.axis] = self.init_actions()  # Uses config to determine actions
            if self.config.args.node_labels and not self.config.args.use_gold_node_labels and \
                    NODE_LABEL_KEY not in labels:
                labels[NODE_LABEL_KEY] = self.init_node_labels()  # Updates self.feature_params
        if self.classifier:  # Already initialized
            pass
        elif self.config.args.classifier == SPARSE:
            from .features.sparse_features import SparseFeatureExtractor
            from .classifiers.linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor(omit_features=self.config.args.omit_features)
            self.classifier = SparsePerceptron(self.config, labels)
        elif self.config.args.classifier == NOOP:
            from .features.empty_features import EmptyFeatureExtractor
            from .classifiers.noop import NoOp
            self.feature_extractor = EmptyFeatureExtractor()
            self.classifier = NoOp(self.config, labels)
        elif self.is_neural_network:
            from .features.dense_features import DenseFeatureExtractor
            from .classifiers.nn.neural_network import NeuralNetwork
            self.feature_extractor = DenseFeatureExtractor(self.feature_params,
                                                           indexed=self.config.args.classifier != MLP,
                                                           hierarchical=self.config.args.classifier == HIERARCHICAL_RNN,
                                                           node_dropout=self.config.args.node_dropout,
                                                           omit_features=self.config.args.omit_features)
            self.classifier = NeuralNetwork(self.config, labels)
        else:
            raise ValueError("Invalid model type: '%s'" % self.config.args.classifier)
        self._update_input_params()

    def set_axis(self, axis, lang):
        if axis is not None:
            self.axis = axis
        if self.axis is None:
            self.axis = self.config.format
        if lang is not None:
            self.lang = lang
        if self.lang is not None:
            suffix = SEPARATOR + self.lang
            if not self.axis.endswith(suffix):
                self.axis += suffix

    @property
    def formats(self):
        return [k.partition(SEPARATOR)[0] for k in self.classifier.labels]

    @property
    def is_neural_network(self):
        return self.config.args.classifier in (MLP, BIRNN, HIGHWAY_RNN, HIERARCHICAL_RNN)

    @property
    def is_retrainable(self):
        return ClassifierProperty.trainable_after_saving in self.classifier_properties

    @property
    def classifier_properties(self):
        return CLASSIFIER_PROPERTIES[self.config.args.classifier]

    @property
    def actions(self):
        return self.classifier.labels[self.axis]

    def init_actions(self):
        return Actions(size=self.config.args.max_action_labels)

    def init_param(self, key):
        if self.feature_extractor:
            self.feature_extractor.init_param(key)

    def init_node_labels(self):
        node_labels = self.feature_params.get(NODE_LABEL_KEY)
        if node_labels is None:
            node_labels = self.node_label_param_def().create_from_config()
            if self.is_neural_network:
                self.feature_params[NODE_LABEL_KEY] = node_labels
        self.init_param(NODE_LABEL_KEY)
        node_labels.init_data()
        return node_labels.data

    def score(self, state, axis):
        features = self.feature_extractor.extract_features(state)
        return self.classifier.score(features, axis=axis), features  # scores is a NumPy array

    def init_features(self, state, train):
        self.init_model()
        axes = [self.axis]
        if self.config.args.node_labels and not self.config.args.use_gold_node_labels:
            axes.append(NODE_LABEL_KEY)
        self.classifier.init_features(self.feature_extractor.init_features(state), axes, train)

    def finalize(self, finished_epoch):
        """
        Copy model, finalizing features (new values will not be added during subsequent use) and classifier (update it)
        :param finished_epoch: whether this is the end of an epoch (or just intermediate checkpoint), for bookkeeping
        :return: a copy of this model with a new feature extractor and classifier (actually classifier may be the same)
        """
        self.config.print("Finalizing model", level=1)
        self.init_model()
        return Model(None, config=self.config.copy(), model=self, is_finalized=True,
                     feature_extractor=self.feature_extractor.finalize(),
                     classifier=self.classifier.finalize(finished_epoch=finished_epoch))

    def save(self, save_init=False):
        """
        Save feature and classifier parameters to files
        """
        if self.filename is not None:
            self.init_model()
            try:
                self.feature_extractor.save(self.filename, save_init=save_init)
                node_labels = self.feature_extractor.params.get(NODE_LABEL_KEY)
                skip_labels = (NODE_LABEL_KEY,) if node_labels and node_labels.size else ()
                self.classifier.save(self.filename, skip_labels=skip_labels,
                                     multilingual=self.config.args.multilingual,
                                     omit_features=self.config.args.omit_features)
                textutil.models["vocab"] = self.config.args.vocab
                save_json(self.filename + ".nlp.json", textutil.models)
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
                self.config.args.multilingual = Classifier.get_property(self.filename, "multilingual")
                self.config.args.omit_features = Classifier.get_property(self.filename, "omit_features")
                self.init_model(init_params=False)
                self.feature_extractor.load(self.filename, order=[p.name for p in self.param_defs()])
                if not is_finalized:
                    self.feature_extractor.unfinalize()
                self._update_input_params()  # Must be before classifier.load() because it uses them to init the model
                self.classifier.load(self.filename)
                self.is_finalized = is_finalized
                self.load_labels()
                try:
                    textutil.models.update(load_json(self.filename + ".nlp.json"))
                    vocab = textutil.models.get("vocab")
                    if vocab:
                        self.config.args.vocab = vocab
                except FileNotFoundError:
                    pass
                self.config.print("\n".join("%s: %s" % i for i in self.feature_params.items()), level=1)
                for param_def in self.param_defs(self.config):
                    param_def.load_to_config(self.feature_extractor.params)
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
            if axis == NODE_LABEL_KEY:  # These are node labels rather than action labels
                node_labels = self.feature_extractor.params.get(NODE_LABEL_KEY)
                if node_labels and node_labels.size:  # Also used for features, so share the dict
                    del all_size
                    labels = node_labels.data
                else:  # Not used as a feature, just get labels
                    labels = UnknownDict() if self.is_finalized else AutoIncrementDict()
                    labels.load(all_size)
            else:  # Action labels for format determined by axis
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
