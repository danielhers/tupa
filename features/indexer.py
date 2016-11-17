from features.feature_extractor_wrapper import FeatureExtractorWrapper


class FeatureIndexer(FeatureExtractorWrapper):
    """
    Wrapper for FeatureEnumerator to replace non-numeric feature values with indices.
    To be used with LSTMNeuralNetwork classifier.
    """
    def __init__(self, feature_extractor, params=None):
        super(FeatureIndexer, self).__init__(feature_extractor, feature_extractor.params if params is None else params)
        if params is None:
            for suffix in "w", "t":
                param = self.params.get(suffix)
                if param is not None:
                    param.indexed = True
        else:
            feature_extractor.params = params

    def extract_features(self, state):
        return self.feature_extractor.extract_features(state)

    def init_features(self, state):
        return self.feature_extractor.init_features(state)

    def filename_suffix(self):
        return self.feature_extractor.filename_suffix()
