import numpy as np

from features.feature_extractor import FeatureExtractor
from parsing.model_util import load_dict, save_dict, UnknownDict, AutoIncrementDict
from parsing.w2v_util import load_word2vec


class FeatureInformation(object):
    def __init__(self, num, dim=1, size=None, init=None, indices=None):
        self.num = num
        self.dim = dim
        self.size = size
        self.init = init
        self.indices = indices


class FeatureIndexer(FeatureExtractor):
    """
    Wrapper for DenseFeatureExtractor to replace non-numeric feature values with indices.
    To be used with NeuralNetwork classifier.
    Initialize with (dimensions, vocabulary_size) pairs as keyword arguments.
    """
    def __init__(self, feature_extractor, feature_types=None, **kwargs):
        self.feature_extractor = feature_extractor
        if feature_types is None:
            self.feature_types = {"numeric": FeatureInformation(feature_extractor.num_features_numeric())}
            for suffix, (dim, size) in kwargs.items():
                if isinstance(dim, int):
                    init = None
                    indices = AutoIncrementDict(size)
                else:
                    w2v = load_word2vec(dim)
                    size = len(w2v.vocab) + 1
                    dim = w2v.vector_size
                    weights = np.array([w2v[x] for x in w2v.vocab])
                    unknown = weights.mean(axis=0)
                    init = (np.vstack((unknown, weights)),)
                    indices = AutoIncrementDict(size, w2v.vocab)
                self.feature_types[suffix] = FeatureInformation(
                    feature_extractor.num_features_non_numeric(suffix), dim, size, init, indices)
        else:
            self.feature_types = feature_types

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> numeric value
        """
        numeric_features, non_numeric_features = self.feature_extractor.extract_features(state)
        features = {"numeric": numeric_features}
        for suffix, values in non_numeric_features:
            indices = self.feature_types[suffix].indices
            features[suffix] = [indices[v] for v in values]
            assert all(isinstance(f, int) for f in features[suffix]),\
                "Invalid feature indices for '%s': %s" % (suffix, features[suffix])
        return features

    def finalize(self):
        feature_types = {s: FeatureInformation(num=f.num, dim=f.dim, size=f.size, init=f.init,
                                               indices=None if f.indices is None else UnknownDict(f.indices))
                         for s, f in self.feature_types.items()}
        return FeatureIndexer(self.feature_extractor, feature_types)

    def save(self, filename):
        d = {s: {"num": f.num, "dim": f.dim, "size": f.size, "init": f.init,
                 "indices": None if f.indices is None else dict(f.indices)}
             for s, f in self.feature_types.items()}
        save_dict(filename + "_features", d)

    def load(self, filename):
        d = load_dict(filename + "_features")
        feature_types = {s: FeatureInformation(num=f["num"], dim=f["dim"], size=f["size"], init=f["init"],
                                               indices=None if f["indices"] is None else UnknownDict(f["indices"]))
                         for s, f in d.items()}
        self.feature_types.update(feature_types)
