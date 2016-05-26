from numbers import Number

import numpy as np

from features.feature_extractor_wrapper import FeatureExtractorWrapper
from features.feature_params import NumericFeatureParameters
from parsing.model_util import AutoIncrementDict
from parsing.w2v_util import load_word2vec


class FeatureIndexer(FeatureExtractorWrapper):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with indices.
    To be used with NeuralNetwork classifier.
    """
    def __init__(self, feature_extractor, params):
        super(FeatureIndexer, self).__init__(feature_extractor, params)

    def init_params(self, param_list):
        params = super(FeatureIndexer, self).init_params(param_list)
        for param in params.values():
            self.init_data(param)
        return params

    def init_data(self, param):
        if param.data is not None or isinstance(param, NumericFeatureParameters):
            return
        param.num = self.feature_extractor.num_features_non_numeric(param.suffix)
        if isinstance(param.dim, Number):
            param.data = AutoIncrementDict(param.size)
        else:
            w2v = load_word2vec(param.dim)
            vocab = w2v.vocab
            if param.size is None or param.size == 0:
                param.size = len(w2v.vocab) + 1
            else:
                vocab = list(vocab)[:param.size - 1]
            param.dim = w2v.vector_size
            weights = np.array([w2v[x] for x in vocab])
            unknown = weights.mean(axis=0)
            param.init = (np.vstack((unknown, weights)),)
            param.data = AutoIncrementDict(param.size, vocab)

    def extract_features(self, state, train):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param train: whether we are in training
        :return dict of feature name -> numeric value
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state, train)
        features = {NumericFeatureParameters.SUFFIX: numeric_features}
        for suffix, values in non_numeric_features:
            param = self.params[suffix]
            features[suffix] = [param.data[v] for v in values]
            assert all(isinstance(f, int) for f in features[suffix]),\
                "Invalid feature indices for '%s': %s" % (suffix, features[suffix])
        return features

    def filename_suffix(self):
        return "_indices"
