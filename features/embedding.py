from collections import defaultdict
from numbers import Number

import numpy as np

from features.feature_extractor import FeatureExtractor
from parsing.config import Config
from parsing.model_util import load_dict, save_dict, UnknownDict, KeyDefaultDict
from parsing.w2v_util import load_word2vec


class FeatureEmbedding(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric features with embeddings
    and return a list of numbers rather than (name, value) pairs.
    To be used with DensePerceptron classifier.
    Initialize with (dimensions,) singletons as keyword arguments.
    """
    def __init__(self, feature_extractor, embedding=None, **kwargs):
        self.feature_extractor = feature_extractor
        self.dims = {s: d[0] if isinstance(d, (list, tuple)) else d for s, d in kwargs.items()}
        self.embedding = KeyDefaultDict(self.init_embedding) if embedding is None else embedding

    def init_embedding(self, suffix):
        dim = self.dims[suffix]
        if isinstance(dim, Number):
            embedding = defaultdict(lambda d=dim: Config().random.normal(size=d))
            embedding[UnknownDict.UNKNOWN]  # Initialize unknown value
            return embedding
        # Otherwise, not a number but a string with path to word vectors file
        w2v = load_word2vec(dim)
        unk = Config().random.normal(size=w2v.vector_size)
        self.dims[suffix] = w2v.vector_size
        return UnknownDict(w2v.vocab, unk)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return vector of concatenated numeric and embedding features
        """
        numeric_features, non_numeric_features = \
            self.feature_extractor.extract_features(state)
        features = [np.array(numeric_features, dtype=float)]
        for suffix, values in non_numeric_features:
            embedding = self.embedding[suffix]
            features += [embedding[v] for v in values]
        assert sum(map(len, features)) == self.num_features(),\
            "Invalid total number of features: %d != %d " % (
                sum(map(len, features)), self.num_features())
        return np.hstack(features).reshape((-1, 1))

    def num_features(self):
        ret = self.feature_extractor.num_features_numeric()
        for suffix in self.dims:
            self.init_embedding(suffix)
            ret += self.dims[suffix] * self.feature_extractor.num_features_non_numeric(suffix)
        return ret

    def finalize(self):
        embedding = {s: UnknownDict(e) for s, e in self.embedding.items()}
        return FeatureEmbedding(self.feature_extractor, embedding, **self.dims)

    def save(self, filename):
        d = {s: dict(e) for s, e in self.embedding.items()}
        save_dict(filename + "_embedding", d)

    def load(self, filename):
        d = load_dict(filename + "_embedding")
        self.embedding.update({s: UnknownDict(e) for s, e in d.items()})
