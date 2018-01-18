from collections import OrderedDict

from .feature_extractor import FeatureExtractor
from .feature_params import FeatureParameters, NumericFeatureParameters
from ..model_util import MISSING_VALUE, UnknownDict, save_dict, load_dict

NON_NUMERIC_FEATURE_SUFFIXES = "wtdencpAT#^$"
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
    # shape, prefix, suffix
    "s0#s1#s2#s3#b0#b1#b2#b3#s0l#s0r#s1l#s1r#s0ll#s0lr#s0rl#s0rr#s1ll#s1lr#s1rl#s1rr#s0L#s0R#s1L#s1R#b0L#b0R#",
    "s0^s1^s2^s3^b0^b1^b2^b3^s0l^s0r^s1l^s1r^s0ll^s0lr^s0rl^s0rr^s1ll^s1lr^s1rl^s1rr^s0L^s0R^s1L^s1R^b0L^b0R^",
    "s0$s1$s2$s3$b0$b1$b2$b3$s0l$s0r$s1l$s1r$s0ll$s0lr$s0rl$s0rr$s1ll$s1lr$s1rl$s1rr$s0L$s0R$s1L$s1R$b0L$b0R$",
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
    def __init__(self, params, indexed):
        super().__init__(FEATURE_TEMPLATES)
        self.numeric_features_template = None
        self.non_numeric_feature_templates = OrderedDict()
        for feature_template in self.feature_templates:
            if feature_template.suffix in NON_NUMERIC_FEATURE_SUFFIXES:
                first_element = feature_template.elements[0]
                for element in feature_template.elements:
                    assert len(element.properties) <= 1, "Non-numeric element with %d properties: %s in feature %s" % (
                        len(element.properties), element, feature_template)
                    assert not element.properties or element.properties == first_element.properties, \
                        "Non-uniform feature template element properties: %s, %s" % (element, first_element)
                existing = self.non_numeric_feature_templates.get(feature_template.suffix)
                assert existing is None, "More than one non-numeric feature with '%s' suffix: %s and %s" % (
                    feature_template.suffix, existing, feature_template)
                self.non_numeric_feature_templates[feature_template.suffix] = feature_template
            else:
                assert self.numeric_features_template is None, "More than one numeric feature template: %s and %s" % (
                    self.numeric_features_template, feature_template)
                self.numeric_features_template = feature_template
        self.indexed = indexed
        if self.indexed:
            self.collapse_features(params, INDEXED_FEATURES)
        self.params = OrderedDict(
            (p.suffix, p) for p in [NumericFeatureParameters(self.numeric_num())] +
            [self.init_param(p) for p in params.values() if p.effective_suffix in self.non_numeric_feature_templates])
    
    def init_param(self, param):
        param.num = self.non_numeric_num(param.effective_suffix)
        if self.indexed and param.suffix in INDEXED_FEATURES:
            param.indexed = True
        return param

    def init_features(self, state, suffix=None):
        features = OrderedDict()
        for suffix, param in self.params.items():
            if param.indexed and param.enabled:
                values = [self.get_prop(None, n, None, None, param.effective_suffix, state) for n in state.terminals]
                assert MISSING_VALUE not in values, "Missing value occurred in feature initialization: '%s'" % suffix
                param.init_data()
                features[suffix] = [param.data[v] for v in values]
        return features

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> list of numeric values
        """
        features = OrderedDict()
        for suffix, param in self.params.items():
            if not param.enabled or not param.dim:
                continue
            if param.indexed and param.copy_from:  # Only need one copy of indices
                copy_from = self.params[param.copy_from]
                if copy_from.enabled and copy_from.dim:
                    continue
            feature_template, default, values = (self.numeric_features_template, 0, [state.node_ratio()]) \
                if param.numeric else (self.non_numeric_feature_templates[param.effective_suffix], MISSING_VALUE, [])
            values += self.calc_feature(feature_template, state, default, param.indexed)
            if not param.numeric and not param.indexed:  # Replace categorical values with their values in data dict
                param.init_data()
                values = [v if v == MISSING_VALUE else param.data[v] for v in values]
            features[suffix] = values
        return features

    def get_all_features(self, indexed=False):
        features = [] if indexed else [self.numeric_features_template]
        for suffix, feature_template in self.non_numeric_feature_templates.items():
            param = (self.params or {}).get(suffix)
            if (not param and not indexed) or (param and param.enabled and indexed == param.indexed):
                features.append(feature_template)
        return ([] if indexed else ["ratio"]) + [str(e) for t in features for e in t.elements]

    def collapse_features(self, params, suffixes):
        suffixes = {p.copy_from if p.external else s for s, p in params.items() if p.dim and s in suffixes}
        if not suffixes:
            return
        longest_suffix = max(suffixes, key=lambda s: len(self.non_numeric_feature_templates[s].elements))
        longest = self.non_numeric_feature_templates[longest_suffix]
        for suffix in suffixes:
            if suffix != longest_suffix:
                template = self.non_numeric_feature_templates.get(suffix)
                if template is not None:
                    template.elements = [e for e in template.elements if e not in longest.elements]

    def numeric_num(self):
        assert self.numeric_features_template is not None, "Missing numeric features template"
        return sum(len(e.properties) for e in self.numeric_features_template.elements) + EXTRA_NUMERIC_FEATURES

    def non_numeric_num(self, suffix):
        feature_template = self.non_numeric_feature_templates.get(suffix)
        assert feature_template is not None, "Missing feature template for suffix '%s'" % suffix
        return sum(len(e.properties) for e in feature_template.elements)

    def finalize(self):
        return type(self)(FeatureParameters.copy(self.params, UnknownDict), self.indexed)

    def unfinalize(self):
        """
        Opposite of finalize(): replace each feature parameter's data dict with a DropoutDict again, to keep training
        """
        for param in self.params.values():
            param.unfinalize()

    def save(self, filename, save_init=True):
        save_dict(filename + FILENAME_SUFFIX, FeatureParameters.copy(self.params, copy_init=save_init))

    def load(self, filename):
        self.params = FeatureParameters.copy(load_dict(filename + FILENAME_SUFFIX), UnknownDict)
        if self.indexed:
            self.collapse_features(self.params, INDEXED_FEATURES)
