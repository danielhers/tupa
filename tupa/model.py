from enum import Enum

from .action import Actions
from .config import Config, SPARSE, MLP_NN, BILSTM_NN, NOOP
from .features.enumerator import FeatureEnumerator
from .features.feature_params import FeatureParameters
from .features.indexer import FeatureIndexer
from .model_util import UnknownDict

ACTION_AXIS = 0
LABEL_AXIS = 1


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


PARAM_DEFS = (
    ParameterDefinition("n", dim="node_label_dim",    size="max_node_labels",    min_count="min_node_label_count"),
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
    def __init__(self, model_type, filename, feature_extractor=None, classifier=None):
        self.args = Config().args
        self.model_type = model_type or SPARSE
        self.filename = filename
        self.actions = self.labels = self.feature_extractor = self.classifier = None
        if feature_extractor or classifier:
            self.feature_extractor = feature_extractor
            self.classifier = classifier
            self.classifier.input_params = self.feature_extractor.params
            self.load_labels()

    def init_model(self, init_params=True):
        if self.classifier:
            return
        values = max_values = feature_params = ()
        if init_params:
            self.actions = Actions()
            max_values = [self.args.max_action_labels]
            values = [self.actions.all]
            feature_params = [p.create_from_config() for p in PARAM_DEFS]
            node_labels = feature_params[0]
            FeatureEnumerator.init_data(node_labels)
            if Config().node_labels and node_labels.size:
                self.labels = node_labels.data
                values.append(self.labels.all)
                max_values.append(self.args.max_node_labels)

        if self.model_type == SPARSE:
            from .features.sparse_features import SparseFeatureExtractor
            from .classifiers.linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor()
            self.classifier = SparsePerceptron(self.filename, values)
        elif self.model_type == MLP_NN:
            from .classifiers.nn.feedforward import MLP
            from .features.dense_features import DenseFeatureExtractor
            self.feature_extractor = FeatureEnumerator(DenseFeatureExtractor(), feature_params)
            self.classifier = MLP(self.filename, values, self.feature_extractor.params, max_num_labels=max_values)
        elif self.model_type == BILSTM_NN:
            from .classifiers.nn.bilstm import BiLSTM
            from .features.dense_features import DenseFeatureExtractor
            self.feature_extractor = FeatureIndexer(FeatureEnumerator(DenseFeatureExtractor(), feature_params))
            self.classifier = BiLSTM(self.filename, values, self.feature_extractor.params, max_num_labels=max_values)
        elif self.model_type == NOOP:
            from .features.empty_features import EmptyFeatureExtractor
            from .classifiers.noop import NoOp
            self.feature_extractor = EmptyFeatureExtractor()
            self.classifier = NoOp(self.filename, values)
        else:
            raise ValueError("Invalid model type: '%s'" % self.model_type)

    def init_features(self, state, train):
        self.init_model()
        self.classifier.init_features(self.feature_extractor.init_features(state), train)

    def finalize(self, finished_epoch):
        self.init_model()
        return Model(model_type=self.model_type, filename=self.filename,
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
                self.classifier.input_params = self.feature_extractor.params
                self.classifier.load()
                self.load_labels()
                for param in PARAM_DEFS:
                    param.load_to_config(self.feature_extractor.params)
                if hasattr(self.classifier, "max_num_labels"):
                    self.args.max_action_labels = self.classifier.max_num_labels[ACTION_AXIS]
                    if len(self.classifier.labels) > 1:
                        self.args.max_node_labels = self.classifier.max_num_labels[LABEL_AXIS]
            except FileNotFoundError:
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e

    def load_labels(self):
        self.init_model()
        self.actions = Actions(self.classifier.labels[ACTION_AXIS])
        if len(self.classifier.labels) > 1:
            node_labels = self.feature_extractor.params.get("n")
            if node_labels is not None and node_labels.size:  # Use same list of node labels as for features
                self.labels = node_labels.data
            else:  # Not used as a feature, just get labels
                self.labels = UnknownDict()
                _, self.labels.all = self.classifier.labels
            self.classifier.labels = (self.actions.all, self.labels.all)
        else:
            self.labels = None
            self.classifier.labels = (self.actions.all,)

    def get_classifier_properties(self):
        return CLASSIFIER_PROPERTIES[self.model_type]
