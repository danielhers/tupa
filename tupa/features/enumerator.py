import numpy as np

from .feature_extractor_wrapper import FeatureExtractorWrapper
from .feature_params import MISSING_VALUE
from .feature_params import NumericFeatureParameters
from ..model_util import DropoutDict


class FeatureEnumerator(FeatureExtractorWrapper):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with numbers.
    To be used with NeuralNetwork classifier.
    """

    def __init__(self, feature_extractor=None, params=None):
        super().__init__(feature_extractor, params)

    def init_params(self, feature_extractor, param_list):
        params = super().init_params(feature_extractor, param_list)
        for param in list(params.values()):
            if feature_extractor.features_exist(param.effective_suffix):
                if not isinstance(param, NumericFeatureParameters):
                    param.num = feature_extractor.num_features_non_numeric(param.effective_suffix)
                    if param.data is None:
                        self.init_data(param)
            else:
                del params[param.suffix]
        return params

    @staticmethod
    def init_data(param):
        keys = ()
        if param.dim and param.external:
            vectors = FeatureExtractorWrapper.get_word_vectors(param)
            keys = vectors.keys()
            param.init = np.array(list(vectors.values()))
        param.data = DropoutDict(size=param.size, keys=keys, dropout=param.dropout, min_count=param.min_count)

    def init_features(self, state, suffix=None):
        param = self.params[suffix]
        assert param.indexed, "Cannot initialize non-indexed parameter '%s'" % suffix
        values = self.feature_extractor.init_features(state, param.effective_suffix)
        assert MISSING_VALUE not in values, "Missing value occurred in feature initialization: '%s'" % suffix
        values = [param.data[v] for v in values]
        assert all(isinstance(f, int) for f in values), "Invalid feature numbers for '%s': %s" % (suffix, values)
        return values

    def extract_features(self, state, params=None):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param params: ignored
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

    def restore(self):
        """
        Opposite of finalize(): replace each feature parameter's data dict with a DropoutDict again, to keep training
        """
        for param in self.params.values():
            param.data = DropoutDict(param.data, size=param.size, dropout=param.dropout, min_count=param.min_count)

    def collapse_features(self, suffixes):
        self.feature_extractor.collapse_features({p.copy_from if p.external else s for s, p in self.params.items()
                                                  if p.dim and s in suffixes})

    def filename_suffix(self):
        return ".enum"
