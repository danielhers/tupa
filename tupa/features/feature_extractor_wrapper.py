from ucca.textutil import get_word_vectors

from .feature_extractor import FeatureExtractor
from .feature_params import copy_params, NumericFeatureParameters
from ..model_util import load_dict, save_dict, UnknownDict


class FeatureExtractorWrapper(FeatureExtractor):
    def __init__(self, feature_extractor, params):
        """
        :param feature_extractor: DenseFeatureExtractor to wrap
        :param params: list of FeatureParameters
        """
        super().__init__(feature_extractor=feature_extractor,
                         params=params if isinstance(params, dict) else self.init_params(feature_extractor, params))

    def init_params(self, feature_extractor, param_list):
        params = {} if param_list is None else {p.suffix: p for p in param_list}
        if feature_extractor:
            param = NumericFeatureParameters(feature_extractor.num_features_numeric())
            params[param.suffix] = param
        return params

    def init_features(self, state, suffix=None):
        return {s: self.feature_extractor.init_features(state, s)
                for s, p in self.params.items() if p.indexed and p.dim}

    def extract_features(self, state, params=None):
        return self.feature_extractor.extract_features(state)

    def finalize(self):
        return type(self)(self.feature_extractor, copy_params(self.params, UnknownDict))

    def restore(self):
        self.feature_extractor.restore()

    def save(self, filename):
        save_dict(filename + self.filename_suffix(), copy_params(self.params))

    def load(self, filename):
        self.params = copy_params(load_dict(filename + self.filename_suffix()), UnknownDict)

    def filename_suffix(self):
        return self.feature_extractor.filename_suffix()

    @staticmethod
    def get_word_vectors(param):
        vectors, param.dim = get_word_vectors(param.dim, param.size, param.filename)
        if param.size is not None:
            assert len(vectors) <= param.size, "Loaded more vectors than requested: %d>%d" % (len(vectors), param.size)
        assert vectors, "Cannot load word vectors. Install them using `python -m spacy download en` or choose a file " \
                        "using the --word-vectors option."
        param.size = len(vectors)
        return vectors
