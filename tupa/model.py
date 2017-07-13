from .action import Actions
from .config import Config, SPARSE, MLP_NN, BILSTM_NN, NOOP
from .features.enumerator import FeatureEnumerator
from .features.feature_params import FeatureParameters
from .features.indexer import FeatureIndexer
from .model_util import UnknownDict

ACTION_AXIS = 0
LABEL_AXIS = 1


class Model(object):
    def __init__(self, model_type, filename, feature_extractor=None, model=None):
        if model_type is None:
            model_type = SPARSE
        self.model_type = model_type
        self.filename = filename
        self.actions = Actions()
        self.labels = None
        if feature_extractor is not None or model is not None:
            self.feature_extractor = feature_extractor
            self.model = model
            self.model.input_params = self.feature_extractor.params
            self.load_labels()
            return

        max_values = [Config().args.max_action_labels]
        values = [self.actions.all]
        node_labels = FeatureParameters("n", Config().args.node_label_dim, Config().args.max_node_labels,
                                        min_count=Config().args.min_node_label_count)
        self.feature_params = [
            node_labels,
            FeatureParameters("W", Config().args.word_dim_external, Config().args.max_words_external,
                              Config().args.word_dropout_external, Config().args.update_word_vectors, copy_from="w",
                              filename=Config().args.word_vectors),
            FeatureParameters("w", Config().args.word_dim, Config().args.max_words, Config().args.word_dropout),
            FeatureParameters("t", Config().args.tag_dim, Config().args.max_tags),
            FeatureParameters("d", Config().args.dep_dim, Config().args.max_deps),
            FeatureParameters("e", Config().args.edge_label_dim, Config().args.max_edge_labels),
            FeatureParameters("p", Config().args.punct_dim, Config().args.max_puncts),
            FeatureParameters("A", Config().args.action_dim, Config().args.max_action_types),
            FeatureParameters("T", Config().args.ner_dim, Config().args.max_ner_types),
        ]
        FeatureEnumerator.init_data(node_labels)
        if Config().node_labels and node_labels.size:
            self.labels = node_labels.data
            values.append(self.labels.all)
            max_values.append(Config().args.max_node_labels)

        if model_type == SPARSE:
            from .features.sparse_features import SparseFeatureExtractor
            from .classifiers.linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor()
            self.model = SparsePerceptron(filename, values)
        elif model_type == MLP_NN:
            from .classifiers.nn.feedforward import MLP
            from .features.dense_features import DenseFeatureExtractor
            self.feature_extractor = FeatureEnumerator(DenseFeatureExtractor(), self.feature_params)
            self.model = MLP(filename, values, self.feature_extractor.params, max_num_labels=max_values)
        elif model_type == BILSTM_NN:
            from .classifiers.nn.bilstm import BiLSTM
            from .features.dense_features import DenseFeatureExtractor
            self.feature_extractor = FeatureIndexer(FeatureEnumerator(DenseFeatureExtractor(), self.feature_params))
            self.model = BiLSTM(filename, values, self.feature_extractor.params, max_num_labels=max_values)
        elif model_type == NOOP:
            from .features.empty_features import EmptyFeatureExtractor
            from .classifiers.noop import NoOp
            self.feature_extractor = EmptyFeatureExtractor()
            self.model = NoOp(filename, values)
        else:
            raise ValueError("Invalid model type: '%s'" % model_type)

    def init_features(self, state, train):
        self.model.init_features(self.feature_extractor.init_features(state), train)

    def finalize(self, finished_epoch):
        return Model(model_type=self.model_type,
                     filename=self.filename,
                     feature_extractor=self.feature_extractor.finalize(),
                     model=self.model.finalize(finished_epoch=finished_epoch))

    def save(self):
        if self.filename is not None:
            try:
                self.feature_extractor.save(self.filename)
                self.model.save()
            except Exception as e:
                raise IOError("Failed saving model to '%s'" % self.filename) from e

    def load(self):
        if self.filename is not None:
            try:
                self.feature_extractor.load(self.filename)
                self.model.load()
                self.model.input_params = self.feature_extractor.params
                self.load_labels()
                self.load_params_to_config()
            except FileNotFoundError:
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e

    def load_params_to_config(self):
        p = self.feature_extractor.params.get("n")
        if p is not None:
            Config().args.node_label_dim = p.dim
            Config().args.max_node_labels = p.size
            Config().args.min_node_label_count = p.min_count
        p = self.feature_extractor.params.get("W")
        if p is not None:
            Config().args.word_dim_external = p.dim
            Config().args.max_words_external = p.size
            Config().args.word_dropout_external = p.dropout
            Config().args.update_word_vectors = p.updated
            Config().args.word_vectors = p.filename
        p = self.feature_extractor.params.get("w")
        if p is not None:
            Config().args.word_dim = p.dim
            Config().args.max_words = p.size
            Config().args.word_dropout = p.dropout
        p = self.feature_extractor.params.get("t")
        if p is not None:
            Config().args.tag_dim = p.dim
            Config().args.max_tags = p.size
        p = self.feature_extractor.params.get("d")
        if p is not None:
            Config().args.dep_dim = p.dim
            Config().args.max_deps = p.size
        p = self.feature_extractor.params.get("e")
        if p is not None:
            Config().args.edge_label_dim = p.dim
            Config().args.max_edge_labels = p.size
        p = self.feature_extractor.params.get("p")
        if p is not None:
            Config().args.punct_dim = p.dim
            Config().args.max_puncts = p.size
        p = self.feature_extractor.params.get("A")
        if p is not None:
            Config().args.action_dim = p.dim
            Config().args.max_action_types = p.size
        p = self.feature_extractor.params.get("T")
        if p is not None:
            Config().args.ner_dim = p.dim
            Config().args.max_ner_types = p.size
        if hasattr(self.model, "max_num_labels"):
            Config().args.max_action_labels = self.model.max_num_labels[ACTION_AXIS]
            if len(self.model.labels) > 1:
                Config().args.max_node_labels = self.model.max_num_labels[LABEL_AXIS]

    def load_labels(self):
        self.actions.all = self.model.labels[ACTION_AXIS]
        if len(self.model.labels) > 1:
            node_labels = self.feature_extractor.params.get("n")
            if node_labels is not None and node_labels.size:  # Use same list of node labels as for features
                self.labels = node_labels.data
            else:  # Not used as a feature, just get labels
                self.labels = UnknownDict()
                _, self.labels.all = self.model.labels
            self.model.labels = (self.actions.all, self.labels.all)
        else:
            self.labels = None
            self.model.labels = (self.actions.all,)
