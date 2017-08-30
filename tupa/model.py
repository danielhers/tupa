from enum import Enum

from .action import Actions
from .config import Config, SPARSE, MLP_NN, BILSTM_NN, NOOP
from .features.enumerator import FeatureEnumerator
from .features.feature_params import FeatureParameters
from .features.indexer import FeatureIndexer
from .model_util import UnknownDict


class ParameterDefinition(object):
    def __init__(self, name, **param_attr_to_config_attr):
        self.name = name
        self.param_attr_to_config_attr = param_attr_to_config_attr

    def create_from_config(self):
        args = Config().args
        return FeatureParameters(self.name, **{k: getattr(args, v) if hasattr(args, v) else v
                                               for k, v in self.param_attr_to_config_attr.items()})

    def load_to_config(self, params):
        param = params.get(self.name)
        if param is not None:
            args = Config().args
            Config().update({v: getattr(param, k) for k, v in self.param_attr_to_config_attr.items()
                             if hasattr(args, v)})

NODE_LABEL_KEY = "n"

PARAM_DEFS = (
    ParameterDefinition(NODE_LABEL_KEY, dim="node_label_dim", size="max_node_labels", min_count="min_node_label_count"),
    ParameterDefinition("c", dim="node_category_dim", size="max_node_categories"),
    ParameterDefinition("W", dim="word_dim_external", size="max_words_external", dropout="word_dropout_external",
                        updated="update_word_vectors", filename="word_vectors",  copy_from="w"),
    ParameterDefinition("w", dim="word_dim",          size="max_words",          dropout="word_dropout"),
    ParameterDefinition("t", dim="tag_dim",           size="max_tags"),
    ParameterDefinition("d", dim="dep_dim",           size="max_deps"),
    ParameterDefinition("e", dim="edge_label_dim",    size="max_edge_labels"),
    ParameterDefinition("p", dim="punct_dim",         size="max_puncts"),
    ParameterDefinition("A", dim="action_dim",        size="max_action_types"),
    ParameterDefinition("T", dim="ner_dim",           size="max_ner_types"),
)


class ClassifierProperty(Enum):
    update_only_on_error = 1
    require_init_features = 2
    trainable_after_saving = 3


CLASSIFIER_PROPERTIES = {
    SPARSE: (ClassifierProperty.update_only_on_error,),
    MLP_NN: (ClassifierProperty.trainable_after_saving,),
    BILSTM_NN: (ClassifierProperty.trainable_after_saving, ClassifierProperty.require_init_features),
    NOOP: (),
}


class Model(object):
    def __init__(self, model_type, filename, *args, **kwargs):
        self.args = Config().args
        self.model_type = model_type or SPARSE
        self.filename = filename
        self.feature_extractor = self.classifier = self.feature_params = None
        if args or kwargs:
            self.restore(*args, **kwargs)

    def init_model(self, init_params=True):
        if self.feature_extractor or self.classifier:
            if self.args.format not in self.classifier.labels:
                self.classifier.labels[self.args.format] = self.init_actions()
            if self.args.node_labels and NODE_LABEL_KEY not in self.classifier.labels:
                self.classifier.labels[NODE_LABEL_KEY] = self.init_node_labels().data
            return
        labels = {}
        if init_params:  # Actually use the config state to initialize the features and hyperparameters, otherwise empty
            labels[self.args.format] = self.init_actions()  # Uses config to determine actions
            self.feature_params = [p.create_from_config() for p in PARAM_DEFS]
            if self.args.node_labels:
                labels[NODE_LABEL_KEY] = self.init_node_labels().data
        if self.model_type in (SPARSE, NOOP):
            if self.model_type == SPARSE:
                from .features.sparse_features import SparseFeatureExtractor as FeatureExtractor
                from .classifiers.linear.sparse_perceptron import SparsePerceptron as Classifier
            else:  # NOOP
                from .features.empty_features import EmptyFeatureExtractor as FeatureExtractor
                from .classifiers.noop import NoOp as Classifier
            self.feature_extractor = FeatureExtractor()
            self.classifier = Classifier(self.filename, labels)
        elif self.model_type in (MLP_NN, BILSTM_NN):
            from .features.dense_features import DenseFeatureExtractor
            self.feature_extractor = FeatureEnumerator(DenseFeatureExtractor(), self.feature_params)
            if self.model_type == MLP_NN:
                from .classifiers.nn.feedforward import MLP as Classifier
            else:  # BILSTM_NN
                self.feature_extractor = FeatureIndexer(self.feature_extractor)
                from .classifiers.nn.bilstm import BiLSTM as Classifier
            self.classifier = Classifier(self.filename, labels)
        else:
            raise ValueError("Invalid model type: '%s'" % self.model_type)
        self._update_input_params()

    def init_actions(self):
        return Actions(size=self.args.max_action_labels)

    def init_node_labels(self):
        node_labels = self.feature_params[0]
        FeatureEnumerator.init_data(node_labels)
        return node_labels

    @property
    def actions(self):
        return self.classifier.labels[self.args.format]

    @property
    def labels(self):
        return self.classifier.labels[NODE_LABEL_KEY]

    def init_features(self, state, train):
        self.init_model()
        self.classifier.init_features(self.feature_extractor.init_features(state), train)

    def finalize(self, finished_epoch):
        self.init_model()
        return Model(None, None, model=self,
                     feature_extractor=self.feature_extractor.finalize(),
                     classifier=self.classifier.finalize(finished_epoch=finished_epoch))

    def save(self):
        if self.filename is not None:
            self.init_model()
            try:
                self.feature_extractor.save(self.filename)
                self.classifier.save()
            except Exception as e:
                raise IOError("Failed saving model to '%s'" % self.filename) from e

    def load(self):
        if self.filename is not None:
            try:
                self.init_model(init_params=False)
                self.feature_extractor.load(self.filename)
                self._update_input_params()  # Must be before classifier.load() because it uses them to init the model
                self.classifier.load()
                self.load_labels()
                for param in PARAM_DEFS:
                    param.load_to_config(self.feature_extractor.params)
            except FileNotFoundError:
                self.feature_extractor = self.classifier = None
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e

    def restore(self, model, feature_extractor=None, classifier=None):
        """
        Set all attributes to a reference to existing model, except labels, which will be copied
        :param model: Model to restore
        :param feature_extractor: optional FeatureExtractor to restore instead of model's
        :param classifier: optional Classifier to restore instead of model's
        """
        self.model_type = model.model_type
        self.filename = model.filename
        self.feature_extractor = feature_extractor or model.feature_extractor
        self.classifier = classifier or model.classifier
        self.feature_params = model.feature_params
        self._update_input_params()
        self.classifier.labels = {a: l.save() for a, l in self.classifier.labels.items()}
        self.load_labels()

    def load_labels(self):
        """
        Copy classifier's labels to create new Actions/UnknownDict objects
        Restoring from a model that was just loaded from file, or called by restore()
        """
        for axis, (all_labels, size) in self.classifier.labels.items():
            if axis == NODE_LABEL_KEY:  # These are node labels rather than action labels
                node_labels = self.feature_extractor.params.get(NODE_LABEL_KEY)
                if node_labels is not None and node_labels.size:  # Also used for features, so share the dict
                    del all_labels, size
                    labels = node_labels.data
                else:  # Not used as a feature, just get labels
                    labels = UnknownDict()
                    labels.all = all_labels  # Same as AutoIncrementDict(keys=all_labels)?
            else:  # Action labels for format determined by axis
                labels = Actions(all_labels, size)
            self.classifier.labels[axis] = labels

    def get_classifier_properties(self):
        return CLASSIFIER_PROPERTIES[self.model_type]

    def _update_input_params(self):
        self.classifier.input_params = self.feature_extractor.params
