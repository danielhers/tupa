from collections import defaultdict
from numbers import Number

import numpy as np

from features.feature_extractor_wrapper import FeatureExtractorWrapper
from features.feature_params import NumericFeatureParameters
from parsing.config import Config
from parsing.model_util import UnknownDict
from parsing.w2v_util import load_word2vec


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
        param.num = self.feature_extractor.num_features_non_numeric(param.suffix)
        if isinstance(param.dim, Number):  # Dimensions given as a number, not as a file to load
            param.data = defaultdict(lambda d=param.dim: Config().random.normal(size=d))
            param.data[UnknownDict.UNKNOWN]  # Initialize unknown value
        else:  # Otherwise, not a number but a string with path to word vectors file
            w2v = load_word2vec(param.dim)
            unk = Config().random.normal(size=w2v.vector_size)
            param.dim = w2v.vector_size
            param.data = UnknownDict({x: w2v[x] for x in w2v.vocab}, unk)

    def extract_features(self, state, train):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param train: whether we are in training
        :return array of concatenated numeric and embedding features
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state, train)
        features = [np.array(numeric_features, dtype=float)]
        for suffix, values in non_numeric_features:
            param = self.params[suffix]
            self.init_data(param)
            features += [param.data[v] for v in values]
        assert sum(map(len, features)) == self.num_features(),\
            "Invalid total number of features: %d != %d " % (
                sum(map(len, features)), self.num_features())
        return np.hstack(features).reshape((-1, 1))

    def num_features(self):
        ret = 0
        for param in self.params.values():
            self.init_data(param)
            ret += param.dim * param.num
        return ret

    def filename_suffix(self):
        return "_embedding"
