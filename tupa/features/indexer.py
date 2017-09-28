from .feature_extractor_wrapper import FeatureExtractorWrapper

INDEXED_FEATURES = "W", "w", "t", "d", "T"  # external + learned word embeddings, POS tags, dep rels, entity type


class FeatureIndexer(FeatureExtractorWrapper):
    """
    Wrapper for FeatureEnumerator to replace non-numeric feature values with indices.
    To be used with BiRNN (NeuralNetwork) classifier.
    """
    def __init__(self, feature_extractor, params=None):
        super().__init__(feature_extractor, feature_extractor.params if params is None else params)
        if params is None:
            for suffix in INDEXED_FEATURES:
                param = self.params.get(suffix)
                if param is not None:
                    param.indexed = True
        else:
            feature_extractor.params = params
        self.feature_extractor.collapse_features(INDEXED_FEATURES)

    def load(self, filename):
        super().load(filename)
        self.feature_extractor.params = self.params
        self.feature_extractor.collapse_features(INDEXED_FEATURES)
