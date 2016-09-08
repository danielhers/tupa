from features.feature_extractor import FeatureExtractor
from features.feature_params import copy_params, NumericFeatureParameters
from parsing.model_util import load_dict, save_dict, UnknownDict


class FeatureExtractorWrapper(FeatureExtractor):
    def __init__(self, feature_extractor, params):
        """
        :param feature_extractor: DenseFeatureExtractor to wrap
        :param params: list of FeatureParameters
        """
        self.feature_extractor = feature_extractor
        self.params = params if isinstance(params, dict) else self.init_params(params)

    def init_params(self, param_list):
        params = {p.suffix: p for p in param_list}
        param = NumericFeatureParameters(self.feature_extractor.num_features_numeric())
        params[param.suffix] = param
        return params

    def extract_features(self, state):
        raise NotImplementedError()

    def finalize(self):
        return self.__class__(self.feature_extractor, copy_params(self.params, UnknownDict))

    def save(self, filename):
        save_dict(filename + self.filename_suffix(), copy_params(self.params))

    def load(self, filename):
        self.params.update(copy_params(load_dict(filename + self.filename_suffix()), UnknownDict))

    def filename_suffix(self):
        raise NotImplementedError()
