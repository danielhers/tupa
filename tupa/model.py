from features.enumerator import FeatureEnumerator
from features.feature_params import FeatureParameters
from features.indexer import FeatureIndexer
from tupa.action import Actions
from tupa.config import Config, SPARSE, MLP_NN, BILSTM_NN
from tupa.model_util import UnknownDict

ACTION_AXIS = 0
LABEL_AXIS = 1


class Model(object):
    def __init__(self, model_type, filename, feature_extractor=None, model=None):
        if model_type is None:
            model_type = SPARSE
        self.model_type = model_type
        self.filename = filename
        if feature_extractor is not None or model is not None:
            self.feature_extractor = feature_extractor
            self.model = model
            self.load_labels()
            return
        max_labels = (Config().args.max_action_labels,)
        labels = (Actions().all,)
        node_labels = FeatureParameters("n", Config().args.node_label_dim, Config().args.max_node_labels,
                                        min_count=Config().args.min_node_label_count)
        FeatureEnumerator.init_data(node_labels)
        if node_labels.size:
            self.labels = node_labels.data
            labels += (self.labels.all,)
            max_labels += (Config().args.max_node_labels,)
        else:
            self.labels = None
        if model_type == SPARSE:
            from features.sparse_features import SparseFeatureExtractor
            from linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor()
            self.model = SparsePerceptron(filename, labels)
        elif model_type == MLP_NN:
            from nn.feedforward import MLP
            self.feature_extractor = self.dense_features_wrapper(node_labels)
            self.model = MLP(filename, labels, input_params=self.feature_extractor.params, max_num_labels=max_labels)
        elif model_type == BILSTM_NN:
            from nn.bilstm import BiLSTM
            self.feature_extractor = FeatureIndexer(self.dense_features_wrapper(node_labels))
            self.model = BiLSTM(filename, labels, input_params=self.feature_extractor.params, max_num_labels=max_labels)
        else:
            raise ValueError("Invalid model type: '%s'" % model_type)

    @staticmethod
    def dense_features_wrapper(*args):
        from features.dense_features import DenseFeatureExtractor
        params = [
            FeatureParameters("W", Config().args.word_dim_external, Config().args.max_words_external,
                              Config().args.word_dropout_external, Config().args.update_word_vectors, copy_from="w",
                              filename=Config().args.word_vectors),
            FeatureParameters("w", Config().args.word_dim, Config().args.max_words, Config().args.word_dropout),
            FeatureParameters("t", Config().args.tag_dim, Config().args.max_tags),
            FeatureParameters("d", Config().args.dep_dim, Config().args.max_deps),
            FeatureParameters("e", Config().args.edge_label_dim, Config().args.max_edge_labels),
            FeatureParameters("p", Config().args.punct_dim, Config().args.max_puncts),
            FeatureParameters("x", Config().args.gap_dim, Config().args.max_gaps),
            FeatureParameters("A", Config().args.action_dim, Config().args.max_action_types),
        ] + list(args)
        return FeatureEnumerator(DenseFeatureExtractor(), params)

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
                Actions().all = self.model.labels[ACTION_AXIS]
                self.load_labels()
            except FileNotFoundError:
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e

    def load_labels(self):
        if len(self.model.labels) > 1:
            node_labels = self.feature_extractor.params.get("n")
            if node_labels is not None and node_labels.size:  # Use same list of node labels as for features
                self.labels = node_labels.data
                self.model.labels = (Actions().all, self.labels.all)
            else:  # Not used as a feature, just get labels
                self.labels = UnknownDict()
                self.labels.all = self.model.labels[LABEL_AXIS]
        else:
            self.labels = None
