from parsing.action import Actions
from parsing.config import Config


class Model(object):
    def __init__(self, model_type=None, labels=None, features=None, model=None):
        self._update_only_on_error = None
        self.model_type = model_type
        if features is not None and model is not None:
            self.features = features
            self.model = model
            return

        if model_type == "sparse":
            from classifiers.sparse_perceptron import SparsePerceptron
            from features.sparse_features import SparseFeatureExtractor
            self.features = SparseFeatureExtractor()
            self.model = SparsePerceptron(labels, min_update=Config().args.minupdate)
        elif model_type == "dense":
            from features.embedding import FeatureEmbedding
            from classifiers.dense_perceptron import DensePerceptron
            self.features = self.dense_features_wrapper(FeatureEmbedding)
            self.model = DensePerceptron(labels, num_features=self.features.num_features())
        elif model_type == "nn":
            from features.indexer import FeatureIndexer
            from classifiers.neural_network import NeuralNetwork
            self.features = self.dense_features_wrapper(FeatureIndexer)
            self.model = NeuralNetwork(labels, inputs=self.features.feature_types,
                                       layers=Config().args.layers,
                                       layer_dim=Config().args.layerdim,
                                       activation=Config().args.activation,
                                       init=Config().args.init,
                                       max_num_labels=Config().args.maxlabels,
                                       batch_size=Config().args.batchsize,
                                       minibatch_size=Config().args.minibatchsize,
                                       nb_epochs=Config().args.nbepochs,
                                       optimizer=Config().args.optimizer,
                                       loss=Config().args.loss
                                       )
        else:
            raise ValueError("Invalid model type: '%s'" % model_type)

    @staticmethod
    def dense_features_wrapper(wrapper):
        from features.dense_features import DenseFeatureExtractor
        return wrapper(DenseFeatureExtractor(),
                       w=(Config().args.wordvectors, 10000),
                       t=(Config().args.tagdim, 100),
                       e=(Config().args.labeldim, 15),
                       p=(Config().args.punctdim, 5),
                       x=(Config().args.gapdim, 3),
                       )

    def extract_features(self, *args, **kwargs):
        return self.features.extract_features(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model.score(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.model.update(*args, **kwargs)

    @property
    def update_only_on_error(self):
        if self._update_only_on_error is None:
            self._update_only_on_error = self.model_type in ("sparse", "dense")
        return self._update_only_on_error

    def finalize(self, *args, **kwargs):
        return Model(model_type=self.model_type,
                     features=self.features.finalize(*args, **kwargs),
                     model=self.model.finalize(*args, **kwargs))

    def save(self, *args, **kwargs):
        self.features.save(*args, **kwargs)
        self.model.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.features.load(*args, **kwargs)
        self.model.load(*args, **kwargs)
        Actions().all = self.model.labels
