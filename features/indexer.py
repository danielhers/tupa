from collections import defaultdict

from gensim.models.word2vec import Word2Vec

from features.feature_extractor import FeatureExtractor


class FeatureInformation(object):
    def __init__(self, num, dim=None, size=None, init=None, indices=None):
        self.num = num
        self.dim = dim
        self.size = size
        self.init = init
        self.indices = indices


class FeatureIndexer(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with indices.
    To be used with NeuralNetwork classifier.
    Initialize with (dimensions, vocabulary_size) pairs as keyword arguments.
    """
    def __init__(self, feature_extractor, **kwargs):
        self.feature_extractor = feature_extractor
        self.feature_types = {"numeric": FeatureInformation(feature_extractor.num_features_numeric())}
        for suffix, (dim, size) in kwargs.items():
            if isinstance(dim, int):
                init = None
                indices = self.auto_increment_dict(size)
            else:
                print("Loading word vectors from '%s'..." % dim)
                w2v = Word2Vec.load_word2vec_format(dim)
                size = len(w2v.vocab) + 1
                dim = w2v.vector_size
                init = (w2v,)
                indices = self.auto_increment_dict(size, w2v.vocab)
            self.feature_types[suffix] = FeatureInformation(
                feature_extractor.num_features_non_numeric(suffix), dim, size, init, indices)

    @staticmethod
    def auto_increment_dict(size, items=()):
        d = defaultdict(lambda: len(d) if len(d) < size else 0)
        d["<UNKNOWN>"] = 0
        for item in items:
            d[item]
        return d

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> numeric value
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state)
        features = {"numeric": numeric_features}
        for suffix, values in non_numeric_features:
            indices = self.feature_types[suffix].indices
            features[suffix] = [indices[v] for v in values]
        return features
