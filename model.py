from features.feature_params import FeatureParameters
from parsing.action import Actions
from parsing.config import Config, SPARSE_PERCEPTRON, DENSE_PERCEPTRON, FEEDFORWARD_NN


class Model(object):
    def __init__(self, model_type, filename, labels=None, feature_extractor=None, model=None):
        if model_type is None:
            model_type = SPARSE_PERCEPTRON
        self.model_type = model_type
        self.filename = filename
        if feature_extractor is not None and model is not None:
            self.feature_extractor = feature_extractor
            self.model = model
            return

        if model_type == SPARSE_PERCEPTRON:
            from features.sparse_features import SparseFeatureExtractor
            from linear.sparse_perceptron import SparsePerceptron
            self.feature_extractor = SparseFeatureExtractor()
            self.model = SparsePerceptron(filename, labels)
        elif model_type == DENSE_PERCEPTRON:
            from features.embedding import FeatureEmbedding
            from linear.dense_perceptron import DensePerceptron
            self.feature_extractor = self.dense_features_wrapper(FeatureEmbedding)
            self.model = DensePerceptron(filename, labels, num_features=self.feature_extractor.num_features())
        elif model_type == FEEDFORWARD_NN:
            from features.enumerator import FeatureEnumerator
            from nn.feedforward import FeedforwardNeuralNetwork
            self.feature_extractor = self.dense_features_wrapper(FeatureEnumerator)
            self.model = FeedforwardNeuralNetwork(filename, labels, input_params=self.feature_extractor.params)
        else:
            raise ValueError("Invalid model type: '%s'" % model_type)

    @staticmethod
    def dense_features_wrapper(wrapper):
        from features.dense_features import DenseFeatureExtractor
        params = [
            FeatureParameters("W", Config().args.word_dim_external, Config().args.max_words_external,
                              Config().args.word_dropout_external, Config().args.update_word_vectors, copy_from="w",
                              filename=Config().args.word_vectors),
            FeatureParameters("w", Config().args.word_dim, Config().args.max_words, Config().args.word_dropout),
            FeatureParameters("t", Config().args.tag_dim, Config().args.max_tags),
            FeatureParameters("d", Config().args.dep_dim, Config().args.max_deps),
            FeatureParameters("e", Config().args.label_dim, Config().args.max_edge_labels),
            FeatureParameters("p", Config().args.punct_dim, Config().args.max_puncts),
            FeatureParameters("x", Config().args.gap_dim, Config().args.max_gaps),
            FeatureParameters("A", Config().args.action_dim, Config().args.max_actions),
        ]
        return wrapper(DenseFeatureExtractor(), params)

    def init_features(self, state, train):
        self.model.init_features(self.feature_extractor.init_features(state), train)

    def finalize(self, finished_epoch):
        return Model(model_type=self.model_type,
                     filename=self.filename,
                     feature_extractor=self.feature_extractor.finalize(),
                     model=self.model.finalize(finished_epoch))

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
                Actions().all = self.model.labels
            except FileNotFoundError:
                raise
            except Exception as e:
                raise IOError("Failed loading model from '%s'" % self.filename) from e
