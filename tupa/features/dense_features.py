from .feature_extractor import FeatureExtractor
from .feature_params import MISSING_VALUE

NON_NUMERIC_FEATURE_SUFFIXES = "wtdencpAT"
FEATURE_TEMPLATES = (
    # words
    "s0ws1ws2ws3w"  # stack
    "b0wb1wb2wb3w"  # buffer
    "s0lws0rws0uws1lws1rws1uw"  # children
    "s0llws0lrws0luws0rlws0rrws0ruws0ulws0urw"
    "s0uuws1llws1lrws1luws1rlws1rrws1ruw"  # grandchildren
    "s0Lws0Rws0Uws1Lws1Rws1Uwb0Lwb0Rwb0Uw",  # parents
    # POS tags
    "s0ts1ts2ts3t"  # stack
    "b0tb1tb2tb3t",  # buffer
    # dependency relations
    "s0ds1ds2ds3d"  # stack
    "b0db1db2db3d",  # buffer
    # edge tags
    "s0es1es2es3e"  # stack
    "s0les0res0ues1les1res1ue"  # children
    "s0lles0lres0lues0rles0rres0rues0ules0ure"
    "s0uues1lles1lres1lues1rles1rres1rue"  # grandchildren
    "s0Ues1Ueb0Ue"  # parents
    "s0b0eb0s0e"  # specific edges
    "a0ea1e",  # past actions edge tags
    # past action labels
    "a0Aa1A",
    # node labels
    "s0ns1ns2ns3n"  # stack
    "b0nb1nb2nb3n"  # buffer
    "s0lns0rns0uns1lns1rns1un"  # children
    "s0llns0lrns0luns0rlns0rrns0runs0ulns0urn"
    "s0uuns1llns1lrns1luns1rlns1rrns1run"  # grandchildren
    "s0Lns0Rns0Uns1Lns1Rns1Unb0Lnb0Rnb0Un",  # parents
    # node label category suffix
    "s0cs1cs2cs3c"  # stack
    "b0cb1cb2cb3c"  # buffer
    "s0lcs0rcs0ucs1lcs1rcs1uc",  # children
    # separators
    "s0ps1p",
    # NER
    "s0Ts1Ts2Ts3T"  # stack
    "b0Tb1Tb2Tb3T",  # buffer
    # numeric
    "s0s1xs1s0xs0b0xb0s0x"
    "s0xs1xs2xs3x"
    "s0s1ds0b0d"
    "s0hqyPCIRNs1hqyNs2hys3hyN"
    "b0hPCIRN",
)
EXTRA_NUMERIC_FEATURES = 2  # bias, node ratio


class DenseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    Requires wrapping by FeatureEnumerator.
    To be used with a NeuralNetwork classifier.
    """
    def __init__(self, feature_templates=FEATURE_TEMPLATES):
        super().__init__(feature_templates=feature_templates)
        self.numeric_features_template = None
        self.non_numeric_feature_templates = []
        for feature_template in self.feature_templates:
            if feature_template.suffix in NON_NUMERIC_FEATURE_SUFFIXES:
                self.non_numeric_feature_templates.append(feature_template)
            else:
                assert self.numeric_features_template is None, "More than one numeric feature template: %s and %s" % (
                        self.numeric_features_template, feature_template)
                self.numeric_features_template = feature_template
        self.non_numeric_by_suffix = {}
        for feature_template in self.non_numeric_feature_templates:
            for element in feature_template.elements:
                assert len(element.properties) <= 1, "Non-numeric element with %d properties: %s in feature %s" % (
                        len(element.properties), element, feature_template)
                if element.properties:
                    assert element.properties == feature_template.elements[0].properties, \
                        "Non-uniform feature template element properties: %s, %s" % (
                            element, feature_template.elements[0])
            assert feature_template.suffix not in self.non_numeric_by_suffix, \
                "More than one non-numeric feature with '%s' suffix: %s and %s" % (
                    feature_template.suffix, self.non_numeric_by_suffix[feature_template.suffix],
                    feature_template)
            self.non_numeric_by_suffix[feature_template.suffix] = feature_template

    def init_features(self, state, suffix=None):
        return [self.get_prop(None, n, None, None, suffix, state) for n in state.terminals]

    def extract_features(self, state, params=None):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param params: dict of FeatureParameters for each suffix
        :return pair: (list of values for all numeric features,
                       list of (suffix, value) pairs for all non-numeric features)
        """
        numeric_features = [1, state.node_ratio()] + self.calc_feature(self.numeric_features_template, state, default=0)
        non_numeric_features = {}
        for f in self.non_numeric_feature_templates:
            if params is None:
                indexed = False
            else:
                param = params.get(f.suffix)
                if param is None:
                    continue
                indexed = param.indexed
            non_numeric_features[f.suffix] = self.calc_feature(
                f, state, default=MISSING_VALUE if indexed else "", indexed=indexed)
        # assert len(numeric_features) == self.num_features_numeric(), \
        #     "Invalid number of numeric features: %d != %d" % (
        #         len(numeric_features), self.num_features_numeric())
        # for value, element in zip(numeric_features, self.numeric_features_template.elements):
        #     assert isinstance(value, Number), \
        #         "Non-numeric value %s for numeric feature element %s" % (value, element)
        # for values, template in zip(non_numeric_features, self.non_numeric_feature_templates):
        #     for value, element in zip(values, template.elements):
        #         assert not isinstance(value, Number), \
        #             "Numeric value %s for non-numeric feature element %s" % (value, element)
        return numeric_features, non_numeric_features

    def collapse_features(self, suffixes):
        if not suffixes:
            return
        longest_suffix = max(suffixes, key=lambda s: len(self.non_numeric_by_suffix[s].elements))
        longest = self.non_numeric_by_suffix[longest_suffix]
        for suffix in suffixes:
            if suffix != longest_suffix:
                template = self.non_numeric_by_suffix.get(suffix)
                if template is not None:
                    template.elements = [e for e in template.elements if e not in longest.elements]

    def num_features_numeric(self):
        assert self.numeric_features_template is not None, "Missing numeric features template"
        return sum(len(e.properties) for e in self.numeric_features_template.elements) + EXTRA_NUMERIC_FEATURES

    def num_features_non_numeric(self, suffix):
        feature_template = self.non_numeric_by_suffix.get(suffix)
        assert feature_template is not None, "Missing feature template for suffix '%s'" % suffix
        return sum(len(e.properties) for e in feature_template.elements)

    def features_exist(self, suffix):
        template = self.numeric_features_template if suffix == "numeric" else self.non_numeric_by_suffix.get(suffix)
        return template is not None
