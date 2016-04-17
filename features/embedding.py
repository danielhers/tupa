from collections import defaultdict

import numpy as np
from gensim.models.word2vec import Word2Vec

from features.feature_extractor import FeatureExtractor
from parsing.config import Config


class Word2VecWrapper(object):
    def __init__(self, w2v, default):
        self.w2v = w2v
        self.default = default

    def __getitem__(self, item):
        return self.w2v[item] if item in self.w2v else self.default


class FeatureEmbedding(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric features with embeddings
    and return a list of numbers rather than (name, value) pairs.
    """
    def __init__(self, feature_extractor, **dims):
        self.feature_extractor = feature_extractor
        self.sizes = {}
        self.embedding = {}
        for suffix, dim in dims.items():
            if isinstance(dim, int):
                self.sizes[suffix] = dim
                self.embedding[suffix] = defaultdict(lambda s=dim:
                                                     Config().random.normal(size=s))
            else:
                print("Loading word vectors from '%s'..." % dim)
                w2v = Word2Vec.load_word2vec_format(dim)
                unk = Config().random.normal(size=w2v.vector_size)
                self.sizes[suffix] = w2v.vector_size
                self.embedding[suffix] = Word2VecWrapper(w2v, unk)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        numeric_features, non_numeric_features = \
            self.feature_extractor.extract_features(state)
        assert len(numeric_features) == self.feature_extractor.num_features_numeric(), \
            "Invalid number of numeric features: %d != %d" % (
                len(numeric_features), self.feature_extractor.num_features_numeric())
        values = [np.array(numeric_features, dtype=float)]
        for suffix, feature_values in non_numeric_features:
            feature_embedding = self.embedding[suffix]
            values += [feature_embedding[v] for v in feature_values]
        assert sum(map(len, values)) == self.num_features(),\
            "Invalid total number of features: %d != %d " % (
                sum(map(len, values)), self.num_features())
        return np.hstack(values)

    def num_features(self):
        return self.feature_extractor.num_features_numeric() + \
            sum(s * self.feature_extractor.num_features_non_numeric(f)
                for f, s in self.sizes.items())
