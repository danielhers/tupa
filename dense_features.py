from features import FeatureExtractor

# The following features are non-numeric and require embedding: wtepx
FEATURE_TEMPLATES = (
    # words
    "s0ws1ws2ws3w"  # stack
    "b0wb1wb2wb3w"  # buffer
    "s0lws0rws1lws1rws1uw"  # children
    "s0llws0lrws0luws0rlws0rrws0ruws0ulws0urw"
    "s0uuws1llws1lrws1luws1rlws1rrws1ruw"  # grandchildren
    "s0pws1pwb0pw"  # parents
    "a0wa1w",  # past actions
    # POS tags
    "s0ts1ts2ts3t"  # stack
    "b0tb1tb2tb3t",  # buffer
    # edge tags
    "s0es1es2es3e"  # stack
    "s0les0res0ues1les1e1s1ue"  # children
    "s0lles0lres0lues0rles0rres0rues0ulws0ure"
    "s0uues1lles1lres1lues1rles1rres1rue"  # grandchildren
    "s0pes1peb0pe"  # parents
    "a0ea1e"  # past actions
    "s0b0eb0s0e",  # specific edges
    # separators
    "s0ps1p",  # types
    "s0qs1q",  # counts
    # disco
    "s0xs1xs2xs3x",  # gap type
    "s0ys1ys2ys3y",  # sum of gap lengths
    # counts
    "s0Ps0C"  # stack
    "b0Pb0C",  # buffer
    # existing edges
    "s0s1s0b0s0",
    # UCCA-specific
    "s0Is0R"  # stack
    "b0Ib0R",  # buffer
)


class DenseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    """
    def __init__(self):
        super(DenseFeatureExtractor, self).__init__(FEATURE_TEMPLATES)

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        features = {  # The hash sign is to avoid embedding match
            "b#": 1,  # Bias
            "n/t#": state.node_ratio(),  # number of nodes divided by number of terminals
        }
        for feature_template in self.feature_templates:
            features[feature_template.name] = self.calc_feature(feature_template, state)
        return features

