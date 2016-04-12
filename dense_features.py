from features import FeatureExtractor

NON_NUMERIC_FEATURE_SUFFIXES = "wtepx"
FEATURE_TEMPLATES = (
    # words
    "s0ws1ws2ws3w"  # stack
    "b0wb1wb2wb3w"  # buffer
    "s0lws0rws0uws1lws1rws1uw"  # children
    "s0llws0lrws0luws0rlws0rrws0ruws0ulws0urw"
    "s0uuws1llws1lrws1luws1rlws1rrws1ruw"  # grandchildren
    "s0pws1pwb0pw"  # parents
    "a0wa1w",  # past actions
    # POS tags
    "s0ts1ts2ts3t"  # stack
    "b0tb1tb2tb3t",  # buffer
    # edge tags
    "s0es1es2es3e"  # stack
    "s0les0res0ues1les1res1ue"  # children
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
        self.numeric_feature_templates = []
        self.non_numeric_feature_templates = []
        for feature_template in self.feature_templates:
            if feature_template.suffix in NON_NUMERIC_FEATURE_SUFFIXES:
                self.non_numeric_feature_templates.append(feature_template)
            else:
                self.numeric_feature_templates.append(feature_template)
        self.non_numeric_feature_templates_by_suffix = {
            f.suffix: f for f in self.non_numeric_feature_templates}

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return pair: (list of values for all numeric features,
                       list of (suffix, value) pairs for all non-numeric features)
        """
        numeric_features = [
           (1,),  # Bias
           (state.node_ratio(),),  # number of nodes divided by number of terminals
        ] + [self.calc_feature(f, state) for f in self.numeric_feature_templates]
        non_numeric_features = [(f.suffix, self.calc_feature(f, state, ""))
                                for f in self.non_numeric_feature_templates]
        return numeric_features, non_numeric_features

    def num_features_numeric(self):
        return sum(len(f.elements) for f in self.numeric_feature_templates)

    def num_features_non_numeric(self, suffix):
        return len(self.non_numeric_feature_templates_by_suffix[suffix].elements)
