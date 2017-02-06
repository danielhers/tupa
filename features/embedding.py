from collections import defaultdict

import numpy as np

from features.feature_extractor_wrapper import FeatureExtractorWrapper
from features.feature_params import MISSING_VALUE
from features.feature_params import NumericFeatureParameters
from parsing.config import Config
from parsing.model_util import UnknownDict


class FeatureEmbedding(FeatureExtractorWrapper):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric features with embeddings
    and return an array of numbers rather than (name, value) pairs.
    To be used with DensePerceptron classifier.
    """
    def __init__(self, feature_extractor, params):
        super(FeatureEmbedding, self).__init__(feature_extractor, params)

    def init_data(self, param):
        if param.data is not None or isinstance(param, NumericFeatureParameters):
            return
        param.num = self.feature_extractor.num_features_non_numeric(param.effective_suffix)
        if param.dim:
            if param.external:
                vectors = self.get_word_vectors(param)
                param.data = UnknownDict(vectors, np.zeros(param.dim))
            else:
                param.data = defaultdict(lambda d=param.dim: Config().random.normal(size=d))
                _ = param.data[UnknownDict.UNKNOWN]  # Initialize unknown value
        param.empty = np.zeros(param.dim, dtype=float)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return array of concatenated numeric and embedding features
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state)
        features = [np.array(numeric_features, dtype=float)]
        for suffix, param in sorted(self.params.items()):
            if param.dim:
                values = non_numeric_features.get(param.effective_suffix)
                if values is not None:
                    self.init_data(param)
                    features += [param.empty if v == MISSING_VALUE else param.data[v] for v in values]
        # assert sum(map(len, features)) == self.num_features(),\
        #     "Invalid total number of features: %d != %d " % (
        #         sum(map(len, features)), self.num_features())
        return np.hstack(features).reshape((-1, 1))

    def num_features(self):
        ret = 0
        for param in self.params.values():
            if param.dim:
                self.init_data(param)
                ret += param.dim * param.num
        return ret

    def filename_suffix(self):
        return "_embedding"
