import numpy as np

from features.feature_extractor_wrapper import FeatureExtractorWrapper
from features.feature_params import MISSING_VALUE
from features.feature_params import NumericFeatureParameters
from parsing.model_util import DropoutDict


class FeatureEnumerator(FeatureExtractorWrapper):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with numbers.
    To be used with NeuralNetwork classifier.
    """

    def __init__(self, feature_extractor, params):
        super(FeatureEnumerator, self).__init__(feature_extractor, params)

    def init_params(self, param_list):
        params = super(FeatureEnumerator, self).init_params(param_list)
        for param in list(params.values()):
            if self.feature_extractor.features_exist(param.effective_suffix):
                self.init_data(param)
            else:
                del params[param.suffix]
        return params

    def init_data(self, param):
        if param.data is not None or isinstance(param, NumericFeatureParameters):
            return
        param.num = self.feature_extractor.num_features_non_numeric(param.effective_suffix)
        keys = ()
        if param.dim and param.external:
            vectors = self.get_word_vectors(param)
            keys = vectors.keys()
            param.size += 1
            param.init = (np.array([np.zeros(param.dim)] + list(vectors.values())),)
        param.data = DropoutDict(max_size=param.size, keys=keys, dropout=param.dropout)

    def init_features(self, state, suffix):
        param = self.params[suffix]
        assert param.indexed, "Cannot initialize non-indexed parameter '%s'" % suffix
        values = self.feature_extractor.init_features(state, param.effective_suffix)
        assert MISSING_VALUE not in values, "Missing value occurred in feature initialization: '%s'" % suffix
        values = [param.data[v] for v in values]
        assert all(isinstance(f, int) for f in values), "Invalid feature numbers for '%s': %s" % (suffix, values)
        return values

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> numeric value
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state, self.params)
        features = {NumericFeatureParameters.SUFFIX: numeric_features}
        for suffix, param in self.params.items():
            if param.dim and (param.copy_from is None or not self.params[param.copy_from].dim or not param.indexed):
                values = non_numeric_features.get(param.effective_suffix)
                if values is not None:
                    features[suffix] = values if param.indexed else \
                        [v if v == MISSING_VALUE else param.data[v] for v in values]
                    # assert all(isinstance(f, int) for f in features[suffix]),\
                    #     "Invalid feature numbers for '%s': %s" % (suffix, features[suffix])
        return features

    def collapse_features(self, suffixes):
        self.feature_extractor.collapse_features({self.params[s].copy_from if self.params[s].external else s
                                                  for s in suffixes if self.params[s].dim})

    def filename_suffix(self):
        return "_enum"
