from .feature_extractor import FeatureExtractor


class EmptyFeatureExtractor(FeatureExtractor):

    def __init__(self):
        super().__init__()

    def extract_features(self, state):
        return {}
