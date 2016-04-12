from collections import defaultdict

import numpy as np

from features import FeatureExtractor


class FeatureEmbedding(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric features with embeddings
    and return a list of numbers rather than (name, value) pairs.
    """
    def __init__(self, feature_extractor, **kwargs):
        self.feature_extractor = feature_extractor
        self.sizes = kwargs
        self.embedding = {f: defaultdict(lambda: np.random.normal(size=s))
                          for f, s in self.sizes.items()}

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        numeric_features, non_numeric_features = \
            self.feature_extractor.extract_features(state)
        embedding = []
        for suffix, values in non_numeric_features:
            feature_embedding = self.embedding[suffix]
            embedding += [feature_embedding[v] for v in values]
        return np.ravel((map(float, numeric_features), embedding))

    def num_features(self):
        return self.feature_extractor.num_features_numeric() + \
            sum(s * self.feature_extractor.num_features_non_numeric(f)
                for f, s in self.sizes.items())
