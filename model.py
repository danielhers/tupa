from parsing.action import Actions
from parsing.config import Config


class Model(object):
    def __init__(self, model_type=None, labels=None, features=None, model=None):
        if model_type is None or labels is None:
            assert features is not None
            assert model is not None
            self.features = features
            self.model = model
            return

        if model_type == "sparse":
            from classifiers.sparse_perceptron import SparsePerceptron
            from features.sparse_features import SparseFeatureExtractor
            self.features = SparseFeatureExtractor()
            self.model = SparsePerceptron(labels, min_update=Config().min_update)
        elif model_type == "dense":
            from features.embedding import FeatureEmbedding
            from classifiers.dense_perceptron import DensePerceptron
            self.features = self.dense_features_wrapper(FeatureEmbedding)
            self.model = DensePerceptron(labels, num_features=self.features.num_features())
        elif model_type == "nn":
            from features.indexer import FeatureIndexer
            from classifiers.neural_network import NeuralNetwork
            self.features = self.dense_features_wrapper(FeatureIndexer)
            self.model = NeuralNetwork(labels, inputs=self.features.feature_types)
        else:
            raise ValueError("Invalid model type: '%s'" % model_type)

    @staticmethod
    def dense_features_wrapper(wrapper):
        from features.dense_features import DenseFeatureExtractor
        return wrapper(DenseFeatureExtractor(),
                       w=(Config().word_vectors, 10000),
                       t=(Config().tag_dim, 100),
                       e=(Config().label_dim, 15),
                       p=(Config().punct_dim, 5),
                       x=(Config().gap_dim, 3),
                       )

    def extract_features(self, *args, **kwargs):
        return self.features.extract_features(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model.score(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.model.update(*args, **kwargs)

    def finalize(self, *args, **kwargs):
        return Model(features=self.features.finalize(*args, **kwargs),
                     model=self.model.finalize(*args, **kwargs))

    def save(self, *args, **kwargs):
        self.features.save(*args, **kwargs)
        self.model.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.features.load(*args, **kwargs)
        self.model.load(*args, **kwargs)
        Actions().all = self.model.labels
