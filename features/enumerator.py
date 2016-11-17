import numpy as np

from features.feature_extractor_wrapper import FeatureExtractorWrapper
from features.feature_params import NumericFeatureParameters
from parsing.model_util import DropoutDict
from parsing.w2v_util import load_word2vec


class FeatureEnumerator(FeatureExtractorWrapper):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with numbers.
    To be used with NeuralNetwork classifier.
    """

    def __init__(self, feature_extractor, params):
        super(FeatureEnumerator, self).__init__(feature_extractor, params)

    def init_params(self, param_list):
        params = super(FeatureEnumerator, self).init_params(param_list)
        for suffix, param in list(params.items()):
            if self.feature_extractor.features_exist(suffix):
                self.init_data(param)
            else:
                del params[suffix]
        return params

    def init_data(self, param):
        if param.data is not None or isinstance(param, NumericFeatureParameters):
            return
        param.num = self.feature_extractor.num_features_non_numeric(param.suffix)
        vocab = ()
        try:
            param.dim = int(param.dim)
        except ValueError | TypeError:
            w2v = load_word2vec(param.dim)
            vocab = w2v.vocab
            if param.size is None or param.size == 0:
                param.size = len(w2v.vocab) + 1
            else:
                vocab = list(vocab)[:param.size - 1]
            param.dim = w2v.vector_size
            weights = np.array([w2v[x] for x in vocab])
            unknown = weights.mean(axis=0)
            param.init = np.vstack((unknown, weights))
        param.data = DropoutDict(max_size=param.size, keys=vocab, dropout=param.dropout)

    def init_features(self, state):
        """
        Calculates feature values for all items in initial state
        :param state: initial state of the parser
        :return dict of property name -> list of values
        """
        features = {}
        for suffix, values in self.feature_extractor.init_features(state).items():
            if values is None:
                continue
            param = self.params[suffix]
            if not param.indexed:
                continue
            features[suffix] = [param.data[v] for v in values]
            assert all(isinstance(f, int) for f in features[suffix]),\
                "Invalid feature numbers for '%s': %s" % (suffix, features[suffix])
        return features

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> numeric value
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state, self.params)
        features = {NumericFeatureParameters.SUFFIX: numeric_features}
        for suffix, values in non_numeric_features:
            param = self.params[suffix]
            features[suffix] = values if param.indexed else [param.data[v] for v in values]
            # assert all(isinstance(f, int) for f in features[suffix]),\
            #     "Invalid feature numbers for '%s': %s" % (suffix, features[suffix])
        return features

    def filename_suffix(self):
        return "_enum"
