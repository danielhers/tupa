from features.feature_extractor import FeatureExtractor
from features.feature_params import copy_params, NumericFeatureParameters
from parsing.model_util import load_dict, save_dict, UnknownDict
from ucca.textutil import get_word_vectors


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

    def init_features(self, state):
        return {s: self.feature_extractor.init_features(state, s)
                for s, p in self.params.items() if p.indexed and p.dim}

    def extract_features(self, state):
        return self.feature_extractor.extract_features(state)

    def finalize(self):
        return self.__class__(self.feature_extractor, copy_params(self.params, UnknownDict))

    def save(self, filename):
        save_dict(filename + self.filename_suffix(), copy_params(self.params))

    def load(self, filename):
        self.params.update(copy_params(load_dict(filename + self.filename_suffix()), UnknownDict))

    def filename_suffix(self):
        return self.feature_extractor.filename_suffix()

    @staticmethod
    def get_word_vectors(param):
        vectors, param.dim = get_word_vectors(param.dim, param.size, param.filename)
        if param.size is None:
            param.size = len(vectors)
        else:
            assert len(vectors) == param.size, "Number of loaded vectors differs from requested: %d != %d" % (
                len(vectors), param.size)
        return vectors
