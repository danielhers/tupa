from collections import defaultdict

import numpy as np

from features.feature_extractor import FeatureExtractor


class FeatureEmbedding(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric features with embeddings
    and return a list of numbers rather than (name, value) pairs.
    """
    def __init__(self, feature_extractor, **kwargs):
        self.feature_extractor = feature_extractor
        self.sizes = kwargs
        self.embedding = {f: defaultdict(self.vector_creator(s))
                          for f, s in self.sizes.items()}

    @staticmethod
    def vector_creator(s):
        return lambda: np.random.normal(size=s)

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
