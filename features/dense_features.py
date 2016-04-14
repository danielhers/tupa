from features.feature_extractor import FeatureExtractor

NON_NUMERIC_FEATURE_SUFFIXES = "wtepx"
FEATURE_TEMPLATES = (
    # words
    "s0ws1ws2ws3w"  # stack
    "b0wb1wb2wb3w"  # buffer
    "s0lws0rws0uws1lws1rws1uw"  # children
    "s0llws0lrws0luws0rlws0rrws0ruws0ulws0urw"
    "s0uuws1llws1lrws1luws1rlws1rrws1ruw"  # grandchildren
    "s0Uws1Uwb0Uw"  # parents
    "a0wa1w",  # past actions
    # POS tags
    "s0ts1ts2ts3t"  # stack
    "b0tb1tb2tb3t",  # buffer
    # edge tags
    "s0es1es2es3e"  # stack
    "s0les0res0ues1les1res1ue"  # children
    "s0lles0lres0lues0rles0rres0rues0ulws0ure"
    "s0uues1lles1lres1lues1rles1rres1rue"  # grandchildren
    "s0Ues1Ueb0Ue"  # parents
    "a0ea1e"  # past actions
    "s0b0eb0s0e",  # specific edges
    # separators
    "s0ps1p",
    # gap types
    "s0xs1xs2xs3x",
    # numeric
    "s0qyPCIRs1qys2ys3y"
    "b0PCIR"
    "s0s1s0b0s0"
)
EXTRA_NUMERIC_FEATURES = 2  # bias, node ratio


class DenseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    """
    def __init__(self):
        super(DenseFeatureExtractor, self).__init__(FEATURE_TEMPLATES)
        self.numeric_features_template = None
        self.non_numeric_feature_templates = []
        for feature_template in self.feature_templates:
            if feature_template.suffix in NON_NUMERIC_FEATURE_SUFFIXES:
                self.non_numeric_feature_templates.append(feature_template)
            else:
                assert self.numeric_features_template is None, \
                    "More than one numeric feature template: %s and %s" % (
                        self.numeric_features_template, feature_template)
                self.numeric_features_template = feature_template
        self.non_numeric_by_suffix = {}
        for feature_template in self.non_numeric_feature_templates:
            for element in feature_template.elements:
                assert len(element.properties) <= 1,\
                    "Non-numeric element with %d properties: %s in feature %s" % (
                        len(element.properties), element, feature_template)
            assert feature_template.suffix not in self.non_numeric_by_suffix, \
                "More than one non-numeric feature with '%s' suffix: %s and %s" % (
                    feature_template.suffix,
                    self.non_numeric_by_suffix[feature_template.suffix],
                    feature_template)
            self.non_numeric_by_suffix[feature_template.suffix] = feature_template

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return pair: (list of values for all numeric features,
                       list of (suffix, value) pairs for all non-numeric features)
        """
        numeric_features = [1, state.node_ratio()] + \
            self.calc_feature(self.numeric_features_template, state, default=-1)
        non_numeric_features = [(f.suffix, self.calc_feature(f, state, default=""))
                                for f in self.non_numeric_feature_templates]
        return numeric_features, non_numeric_features

    def num_features_numeric(self):
        assert self.numeric_features_template is not None, \
            "Missing numeric features template"
        return sum(len(e.properties) for e in self.numeric_features_template.elements) + \
               len([e for e in self.numeric_features_template.elements
                    if not e.properties]) - 1 + EXTRA_NUMERIC_FEATURES

    def num_features_non_numeric(self, suffix):
        feature_template = self.non_numeric_by_suffix.get(suffix)
        assert feature_template is not None, \
            "Missing feature template for suffix '%s'" % suffix
        return sum(len(e.properties) for e in feature_template.elements)
