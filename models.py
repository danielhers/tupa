from parsing.config import Config


def create_model(model_type, labels):
    if model_type == "sparse":
        from classifiers.sparse_perceptron import SparsePerceptron
        from features.sparse_features import SparseFeatureExtractor
        features = SparseFeatureExtractor()
        model = SparsePerceptron(labels, min_update=Config().min_update)
    elif model_type == "dense":
        from features.embedding import FeatureEmbedding
        from classifiers.dense_perceptron import DensePerceptron
        features = dense_features_wrapper(FeatureEmbedding)
        model = DensePerceptron(labels, num_features=features.num_features())
    elif model_type == "nn":
        from features.indexer import FeatureIndexer
        from classifiers.neural_network import NeuralNetwork
        features = dense_features_wrapper(FeatureIndexer)
        model = NeuralNetwork(labels, inputs=features.feature_types)
    else:
        raise ValueError("Invalid model type: '%s'" % model_type)
    return features, model


def dense_features_wrapper(wrapper):
    from features.dense_features import DenseFeatureExtractor
    return wrapper(DenseFeatureExtractor(),
                   w=(Config().word_vectors, 10000),
                   t=(Config().tag_dim, 100),
                   e=(Config().label_dim, 15),
                   p=(Config().punct_dim, 5),
                   x=(Config().gap_dim, 3),
                   )

