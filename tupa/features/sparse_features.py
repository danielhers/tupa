from .feature_extractor import FeatureExtractor

FEATURE_TEMPLATES = (
    # unigrams (Zhang and Clark 2009):
    "s0tde", "s0we", "s0#e", "s0^e", "s0$e", "s1tde", "s1we", "s1#e", "s1^e", "s1$e",
    "s2tde", "s2we", "s2#e", "s2^e", "s2$e", "s3tde", "s3we", "s3#e", "s3^e", "s3$e",
    "b0wtd", "b1wtd", "b2wtd", "b3wtd", "s0lwe", "s0rwe", "s1lwe", "s1rwe",
    # bigrams (Zhang and Clark 2009):
    "s0ws1w", "s0ws1e", "s0es1w", "s0es1e", "s0wb0w", "s0wb0td",
    "s0eb0w", "s0eb0td", "s1wb0w", "s1wb0td", "s1eb0w", "s1eb0td", "b0wb1w", "b0wb1td", "b0tdb1w", "b0tdb1td",
    # trigrams (Zhang and Clark 2009):
    "s0es1es2#", "s0es1es2e", "s0es1es2e", "s0es1eb0#", "s0es1eb0td",
    "s0es1#b0#", "s0es1#b0td", "s0#s1es2e", "s0#s1eb0td",
    # extended (Zhu et al. 2013):
    "s0llwe", "s0lrwe", "s0rlwe", "s0rrwe", "s1llwe", "s1lrwe", "s1rlwe", "s1rrwe",
    # parents:
    "s0Lwe", "s0Rwe", "s1Lwe", "s1Rwe", "b0Lwe", "b0Rwe",
    # separator (Zhu et al. 2013):
    "s0wp", "s0wep", "s0wq", "s0weq", "s0es1ep", "s0es1eq", "s1wp", "s1wep", "s1wq", "s1weq",
    # disco, unigrams (Maier 2015):
    "s0xwe", "s1xwe", "s2xwe", "s3xwe", "s0xtde", "s1xtde", "s2xtde", "s3xtde", "s0xy", "s1xy", "s2xy", "s3xy",
    # disco, bigrams (Maier 2015):
    "s0xs1e", "s0xs1w", "s0xs1x", "s0ws1x", "s0es1x", "s0xs2e", "s0xs2w", "s0xs2x", "s0es2x",
    "s0ys1y", "s0ys2y", "s0xb0td", "s0xb0w",
    # counts (Tokgöz and Eryiğit 2015):
    "s0P", "s0C", "s0wP", "s0wC", "b0P", "b0C", "b0wP", "b0wC",
    # existing edges (Tokgöz and Eryiğit 2015):
    "s0s1x", "s1s0x", "s0b0x", "b0s0x", "s0b0e", "b0s0e",
    # dependency distance:
    "s0s1d", "s0b0d",
    # past actions (Tokgöz and Eryiğit 2015):
    "a0Ae", "a1Ae",
    # implicit and remote
    "s0I", "s0E", "s0M", "s0wI", "s0wE", "s0wM", "b0I", "b0E", "b0M", "b0wI", "b0wE", "b0wM",
    # node labels
    "s0n", "s0wn", "s0c", "b0n", "b0wn", "b0c", "s0Ln", "s0Rn", "s1Ln", "s1Rn", "b0Ln", "b0Rn",
    # height
    "s0h", "s1h", "b0h", "b1h",
    # NER
    "s0NT", "s1NT", "b0NT", "b1NT",
)


class SparseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    To be used with SparsePerceptron classifier.
    """

    def __init__(self, omit_features=None):
        super().__init__(feature_templates=FEATURE_TEMPLATES, omit_features=omit_features)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> value
        """
        features = {
            "b": 1,  # Bias
            "n/t": state.node_ratio(),  # number of nodes divided by number of terminals
        }
        for feature_template in self.feature_templates:
            values = feature_template.extract(state)
            if values:
                features["%s=%s" % (feature_template.name, " ".join(map(str, values)))] = 1
        return features
