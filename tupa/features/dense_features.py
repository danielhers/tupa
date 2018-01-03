from .feature_extractor import FeatureExtractor
from .feature_params import FeatureParameters, NumericFeatureParameters
from ..model_util import MISSING_VALUE, UnknownDict, save_dict, load_dict

NON_NUMERIC_FEATURE_SUFFIXES = "wtdencpAT"
FEATURE_TEMPLATES = (
    # words
    "s0ws1ws2ws3w"  # stack
    "b0wb1wb2wb3w"  # buffer
    "s0lws0rws1lws1rw"  # children
    "s0llws0lrws0rlws0rrw"
    "s1llws1lrws1rlws1rrw"  # grandchildren
    "s0Lws0Rws1Lws1Rwb0Lwb0Rw",  # parents
    # POS tags
    "s0ts1ts2ts3t"  # stack
    "b0tb1tb2tb3t",  # buffer
    # dependency relations
    "s0ds1ds2ds3d"  # stack
    "b0db1db2db3d",  # buffer
    # edge tags
    "s0es1es2es3e"  # stack
    "s0les0res1les1re"  # children
    "s0lles0lres0rles0rre"
    "s1lles1lres1rles1rre"  # grandchildren
    "s0Les0Res1Les1Reb0Leb0Re"  # parents
    "s0b0eb0s0e"  # specific edges
    "a0ea1e",  # past actions edge tags
    # past action labels
    "a0Aa1A",
    # node labels
    "s0ns1ns2ns3n"  # stack
    "b0nb1nb2nb3n"  # buffer
    "s0lns0rns1lns1rn"  # children
    "s0llns0lrns0rlns0rrn"
    "s1llns1lrns1rlns1rrn"  # grandchildren
    "s0Lns0Rns1Lns1Rnb0Lnb0Rn",  # parents
    # node label category suffix
    "s0cs1cs2cs3c"  # stack
    "b0cb1cb2cb3c"  # buffer
    "s0lcs0rcs1lcs1rc",  # children
    # separators
    "s0p",
    # NER
    "s0Ts1Ts2Ts3T"  # stack
    "b0Tb1Tb2Tb3T",  # buffer
    # numeric
    "s0s1xds1s0xs0b0xdb0s0x"
    "s0xhqyPCIRNs1xhyNs2xhys3xhyN"
    "b0hPCIRN",
)
EXTRA_NUMERIC_FEATURES = 1  # node ratio
INDEXED_FEATURES = "WwtdT"  # external + learned word embeddings, POS tags, dep rels, entity type
FILENAME_SUFFIX = ".enum"


class DenseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    To be used with a NeuralNetwork classifier.
    """
    def __init__(self, params, indexed, feature_templates=FEATURE_TEMPLATES):
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
        self.params = params
        self.indexed = indexed
        if self.indexed:
            self.collapse_features(INDEXED_FEATURES)
        self.params = {p.suffix: p for p in [self.init_param(p) for p in self.params.values()
                                             if p.effective_suffix in self] +
                       [NumericFeatureParameters(self.numeric_num())]}

    def init_param(self, param):
        param.num = self.non_numeric_num(param.effective_suffix)
        if self.indexed and param.suffix in INDEXED_FEATURES:
            param.indexed = True
        return param

    def init_features(self, state, suffix=None):
        features = {}
        for suffix, param in self.params.items():
            if param.indexed and param.enabled:
                values = [self.get_prop(None, n, None, None, param.effective_suffix, state) for n in state.terminals]
                assert MISSING_VALUE not in values, "Missing value occurred in feature initialization: '%s'" % suffix
                param.init_data()
                features[suffix] = [param.data[v] for v in values]
        return features

    def extract_features(self, state, params=None):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param params: ignored
        :return dict of feature name -> numeric value
        """
        numeric_features = [state.node_ratio()] + self.calc_feature(self.numeric_features_template, state, default=0)
        non_numeric_features = {f.suffix: self.calc_feature(f, state, MISSING_VALUE, indexed)
                                for indexed in (False, True) for f in self.get_enabled_features(indexed)}
        # assert len(numeric_features) == self.numeric_num(), \
        #     "Invalid number of numeric features: %d != %d" % (
        #         len(numeric_features), self.numeric_num())
        # for value, element in zip(numeric_features, self.numeric_features_template.elements):
        #     assert isinstance(value, Number), \
        #         "Non-numeric value %s for numeric feature element %s" % (value, element)
        # for values, template in zip(non_numeric_features, self.non_numeric_feature_templates):
        #     for value, element in zip(values, template.elements):
        #         assert not isinstance(value, Number), \
        #             "Numeric value %s for non-numeric feature element %s" % (value, element)
        features = {NumericFeatureParameters.SUFFIX: numeric_features}
        for suffix, param in self.params.items():
            if param.dim and (param.copy_from is None or not self.params[param.copy_from].dim or not param.indexed):
                values = non_numeric_features.get(param.effective_suffix)
                if values is not None:
                    param.init_data()
                    features[suffix] = values if param.indexed else \
                        [v if v == MISSING_VALUE else param.data[v] for v in values]
                    # assert all(isinstance(f, int) for f in features[suffix]),\
                    #     "Invalid feature numbers for '%s': %s" % (suffix, features[suffix])
        return features

    def get_enabled_features(self, indexed=False):
        if self.params is None:
            yield from [] if indexed else self.non_numeric_feature_templates
        for f in self.non_numeric_feature_templates:
            param = self.params.get(f.suffix)
            if param and param.enabled and indexed == param.indexed:
                yield f

    def get_all_features(self, indexed=False):
        return [str(e) for t in self.get_enabled_features(indexed=True) for e in t.elements] if indexed else \
            ["ratio"] + [str(e) for t in [self.numeric_features_template] +
                         list(self.get_enabled_features()) for e in t.elements]

    def collapse_features(self, suffixes):
        suffixes = {p.copy_from if p.external else s for s, p in self.params.items() if p.dim and s in suffixes}
        if not suffixes:
            return
        longest_suffix = max(suffixes, key=lambda s: len(self.non_numeric_by_suffix[s].elements))
        longest = self.non_numeric_by_suffix[longest_suffix]
        for suffix in suffixes:
            if suffix != longest_suffix:
                template = self.non_numeric_by_suffix.get(suffix)
                if template is not None:
                    template.elements = [e for e in template.elements if e not in longest.elements]

    def numeric_num(self):
        assert self.numeric_features_template is not None, "Missing numeric features template"
        return sum(len(e.properties) for e in self.numeric_features_template.elements) + EXTRA_NUMERIC_FEATURES

    def non_numeric_num(self, suffix):
        feature_template = self.non_numeric_by_suffix.get(suffix)
        assert feature_template is not None, "Missing feature template for suffix '%s'" % suffix
        return sum(len(e.properties) for e in feature_template.elements)

    def __contains__(self, suffix):
        return suffix in self.non_numeric_by_suffix

    def finalize(self):
        return type(self)(FeatureParameters.copy(self.params, UnknownDict), self.indexed)

    def restore(self):
        """
        Opposite of finalize(): replace each feature parameter's data dict with a DropoutDict again, to keep training
        """
        for param in self.params.values():
            param.restore()

    def save(self, filename):
        save_dict(filename + FILENAME_SUFFIX, FeatureParameters.copy(self.params))

    def load(self, filename):
        self.params = FeatureParameters.copy(load_dict(filename + FILENAME_SUFFIX), UnknownDict)
        if self.indexed:
            self.collapse_features(INDEXED_FEATURES)
