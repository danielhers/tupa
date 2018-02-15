from collections import OrderedDict
from enum import Enum

from .action import Actions
from .classifiers.classifier import Classifier
from .config import Config, SPARSE, MLP, BIRNN, HIGHWAY_RNN, NOOP
from .features.feature_params import FeatureParameters
from .model_util import UnknownDict, AutoIncrementDict, remove_backup


class ParameterDefinition:
    def __init__(self, config, name, **param_attr_to_arg):
        self.config = config
        self.name = name
        self.param_attr_to_arg = param_attr_to_arg

    @property
    def dim_arg(self):
        return self.param_attr_to_arg["dim"]

    @property
    def size_arg(self):
        return self.param_attr_to_arg["size"]

    @property
    def enabled(self):
        return bool(getattr(self.config.args, self.dim_arg))
    
    @enabled.setter
    def enabled(self, value):
        if value:
            raise ValueError("Can only disable parameter configuration by setting 'enabled' to False")
        setattr(self.config.args, self.dim_arg, 0)

    def create_from_config(self):
        args = self.config.args
        return FeatureParameters(self.name, **{k: getattr(args, v) if hasattr(args, v) else v
                                               for k, v in self.param_attr_to_arg.items()})

    def load_to_config(self, params):
        param = params.get(self.name)
        self.config.update({self.dim_arg: 0, self.size_arg: 0} if param is None else {v: getattr(param, k) for k, v in
                                                                                      self.param_attr_to_arg.items()
                                                                                      if hasattr(self.config.args, v)})

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s(%s, %s)" % (type(self).__name__, self.name,
                               ", ".join("%s='%s'" % i for i in self.param_attr_to_arg.items()))


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
    NOOP: (ClassifierProperty.trainable_after_saving,),
}


class Model:
    def __init__(self, filename, config=None, *args, **kwargs):
        self.config = config or Config().copy()
        self.node_label_param_def = ParameterDefinition(self.config, NODE_LABEL_KEY, dim="node_label_dim",
                                                        size="max_node_labels", min_count="min_node_label_count",
                                                        dropout="node_label_dropout")
        self.param_defs = (
            self.node_label_param_def,
            ParameterDefinition(self.config, "c", dim="node_category_dim", size="max_node_categories"),
            ParameterDefinition(self.config, "W", dim="word_dim_external", size="max_words_external",
                                dropout="word_dropout_external", updated="update_word_vectors", filename="word_vectors",
                                copy_from="w"),
            ParameterDefinition(self.config, "w", dim="word_dim",          size="max_words", dropout="word_dropout"),
            ParameterDefinition(self.config, "t", dim="tag_dim",           size="max_tags",  dropout="tag_dropout"),
            ParameterDefinition(self.config, "d", dim="dep_dim",           size="max_deps",  dropout="dep_dropout"),
            ParameterDefinition(self.config, "e", dim="edge_label_dim",    size="max_edge_labels"),
            ParameterDefinition(self.config, "p", dim="punct_dim",         size="max_puncts"),
            ParameterDefinition(self.config, "A", dim="action_dim",        size="max_action_types"),
            ParameterDefinition(self.config, "T", dim="ner_dim",           size="max_ner_types"),
            ParameterDefinition(self.config, "#", dim="shape_dim",         size="max_shapes"),
            ParameterDefinition(self.config, "^", dim="prefix_dim",        size="max_prefixes"),
            ParameterDefinition(self.config, "$", dim="suffix_dim",        size="max_suffixes"),
        )
        self.filename = filename
        self.feature_extractor = self.classifier = None
        self.feature_params = OrderedDict()
        self.is_finalized = False
        if args or kwargs:
            self.restore(*args, **kwargs)

    def init_model(self, init_params=True):
        labels = self.classifier.labels if self.classifier else OrderedDict()
        if init_params:  # Actually use the config state to initialize the features and hyperparameters, otherwise empty
            for param_def in self.param_defs:
                param = self.feature_params.get(param_def.name)
                if param:
                    param.enabled = param_def.enabled
                elif self.is_neural_network and param_def.enabled:
                    self.feature_params[param_def.name] = param = param_def.create_from_config()
                    self.init_param(param)
            if self.config.format not in labels:
                labels[self.config.format] = self.init_actions()  # Uses config to determine actions
            if self.config.args.node_labels and not self.config.args.use_gold_node_labels and \
                    NODE_LABEL_KEY not in labels:
                labels[NODE_LABEL_KEY] = self.init_node_labels()  # Updates self.feature_params
        if self.classifier:  # Already initialized
            pass
        elif self.config.args.classifier == SPARSE:
            from .features.sparse_features import SparseFeatureExtractor
            from .classifiers.linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor()
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
                                                           node_dropout=self.config.args.node_dropout)
            self.classifier = NeuralNetwork(self.config, labels)
        else:
            raise ValueError("Invalid model type: '%s'" % self.config.args.classifier)
        self._update_input_params()
    
    @property
    def is_neural_network(self):
        return self.config.args.classifier in (MLP, BIRNN, HIGHWAY_RNN)

    def is_retrainable(self):
        return ClassifierProperty.trainable_after_saving in self.get_classifier_properties()

    def init_actions(self):
        return Actions(size=self.config.args.max_action_labels)

    def init_param(self, param):
        if self.feature_extractor:
            self.feature_extractor.init_param(param)

    def init_node_labels(self):
        node_labels = self.feature_params.get(NODE_LABEL_KEY)
        if node_labels is None:
            node_labels = self.node_label_param_def.create_from_config()
            if self.is_neural_network:
                self.feature_params[NODE_LABEL_KEY] = node_labels
        self.init_param(node_labels)
        node_labels.init_data()
        return node_labels.data

    @property
    def actions(self):
        return self.classifier.labels[self.config.format]

    def init_features(self, state, axes, train):
        self.init_model()
        self.classifier.init_features(self.feature_extractor.init_features(state), axes, train)

    def finalize(self, finished_epoch):
        """
        Copy model, finalizing features (new values will not be added during subsequent use) and classifier (update it)
        :param finished_epoch: whether this is the end of an epoch (or just intermediate checkpoint), for bookkeeping
        :return: a copy of this model with a new feature extractor and classifier (actually classifier may be the same)
        """
        if self.config.args.verbose > 1:
            print("Finalizing model")
        self.init_model()
        return Model(None, model=self, is_finalized=True,
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
                self.classifier.save(self.filename, skip_labels=skip_labels)
                self.config.save(self.filename)
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
                self.config.args.classifier = Classifier.get_model_type(self.filename)
                self.init_model(init_params=False)
                self.feature_extractor.load(self.filename, order=[p.name for p in self.param_defs])
                if not is_finalized:
                    self.feature_extractor.unfinalize()
                self._update_input_params()  # Must be before classifier.load() because it uses them to init the model
                self.classifier.load(self.filename)
                self.is_finalized = is_finalized
                self.load_labels()
                if self.config.args.verbose:
                    print("\n".join("%s: %s" % i for i in self.feature_params.items()))
                for param in self.param_defs:
                    param.load_to_config(self.feature_extractor.params)
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
        if self.config.args.verbose > 1:
            print("Restoring %sfinalized model" % ("" if is_finalized else "non-"))
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

    def get_classifier_properties(self):
        return CLASSIFIER_PROPERTIES[self.config.args.classifier]

    def _update_input_params(self):
        self.feature_params = self.classifier.input_params = self.feature_extractor.params

    def get_all_params(self):
        d = OrderedDict()
        d["features"] = self.feature_extractor.get_all_features()
        d.update(("input_" + s, p.data.all) for s, p in self.feature_extractor.params.items() if p.data)
        d.update(self.classifier.get_all_params())
        return d
