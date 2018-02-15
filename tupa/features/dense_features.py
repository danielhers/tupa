from collections import OrderedDict

from .feature_extractor import FeatureExtractor, calc
from .feature_params import FeatureParameters, NumericFeatureParameters
from ..model_util import UNKNOWN_VALUE, MISSING_VALUE, UnknownDict, save_dict, load_dict

FEATURE_TEMPLATES = (
    "s0s1xd" "s1s0x" "s0b0xd" "b0s0x"  # specific edges
    "s0wtdencpT#^$xhqyPCIEMN"
    "s1wtdencT#^$xhyN"
    "s2wtdencT#^$xhy"
    "s3wtdencT#^$xhyN"
    "b0wtdncT#^$hPCIEMN"
    "b1wtdncT#^$"
    "b2wtdncT#^$"
    "b3wtdncT#^$"
    "s0lwenc#^$"
    "s0rwenc#^$"
    "s1lwenc#^$"
    "s1rwenc#^$"
    "s0llwen#^$"
    "s0lrwen#^$"
    "s0rlwen#^$"
    "s0rrwen#^$"
    "s1llwen#^$"
    "s1lrwen#^$"
    "s1rlwen#^$"
    "s1rrwen#^$"
    "s0Lwen#^$"
    "s0Rwen#^$"
    "s1Lwen#^$"
    "s1Rwen#^$"
    "b0Lwen#^$"
    "b0Rwen#^$"
    "s0b0e" "b0s0e"  # specific edges
    "a0eAa1eA",  # past actions
)
INDEXED = "wtdT"  # words, POS tags, dep rels, entity type
DEFAULT = ()  # intermediate value for missing features
FILENAME_SUFFIX = ".enum"


class DenseFeatureExtractor(FeatureExtractor):
    """
    Object to extract features from the parser state to be used in action classification
    To be used with a NeuralNetwork classifier.
    """
    def __init__(self, params, indexed, node_dropout=0, init_params=True):
        super().__init__(FEATURE_TEMPLATES)
        self.indexed = indexed
        self.node_dropout = node_dropout
        if init_params:
            self.params = OrderedDict((p.suffix, p) for p in [NumericFeatureParameters(1)] + list(params.values()))
            for param in self.params.values():
                self.update_param_indexed(param)
            param_values = self.get_param_values(all_params=True)
            for param in self.params.values():
                param.num = len(param_values[param])
                param.node_dropout = self.node_dropout
        else:
            self.params = params
    
    def init_param(self, param):
        self.update_param_indexed(param)
        param.num = len(self.get_param_values(all_params=True)[param])

    def update_param_indexed(self, param):
        param.indexed = self.indexed and param.effective_suffix in INDEXED

    @property
    def feature_template(self):
        return self.feature_templates[0]

    def init_features(self, state, suffix=None):
        features = OrderedDict()
        for suffix, param in self.params.items():
            if param.indexed and param.enabled:
                values = [calc(n, state, param.effective_suffix) for n in state.terminals]
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
        for param, vs in self.get_param_values(state).items():
            param.init_data()  # Replace categorical values with their values in data dict:
            features[param.suffix] = [(UNKNOWN_VALUE if v == DEFAULT else v) if param.numeric else
                                      (MISSING_VALUE if v == DEFAULT else (v if param.indexed else param.data[v]))
                                      for v in vs]
        return features

    def get_param_values(self, state=None, all_params=False):
        indexed = []
        param_values = OrderedDict()
        values = OrderedDict()
        for suffix, param in self.params.items():
            if param.enabled and param.dim or all_params:
                if param.indexed:
                    if param.copy_from:
                        copy_from = self.params.get(param.copy_from)
                        if copy_from and copy_from.enabled and copy_from.dim and not all_params:
                            continue
                    if param.effective_suffix not in indexed:
                        indexed.append(param.effective_suffix)  # Only need one copy of indices
                param_values[param] = values.setdefault(
                    NumericFeatureParameters.SUFFIX if param.numeric else param.effective_suffix,
                    ([state.node_ratio()] if state else [1] if all_params else []) if param.numeric else [])
        for e, prop, value in self.feature_template.extract(state, DEFAULT, "".join(indexed), as_tuples=True,
                                                            node_dropout=self.node_dropout):
            vs = values.get(NumericFeatureParameters.SUFFIX if e.is_numeric(prop) else prop)
            if vs is not None:
                vs.append(value if state else (e, prop))
        return param_values

    def get_all_features(self):
        return ["".join(self.join_props(vs)) for _, vs in sorted(self.get_param_values().items(),
                                                                 key=lambda x: x[0].suffix)]

    @staticmethod
    def join_props(values):
        prev = None
        ret = []
        for element, prop in values:
            prefix = "" if element.is_numeric(prop) and prev == element.str else element.str
            ret.append(prefix + prop)
            prev = element.str
        return ret

    def finalize(self):
        return type(self)(FeatureParameters.copy(self.params, UnknownDict), self.indexed, init_params=False)

    def unfinalize(self):
        """Undo finalize(): replace each feature parameter's data dict with a DropoutDict again, to keep training"""
        for param in self.params.values():
            param.unfinalize()
            self.node_dropout = param.node_dropout

    def save(self, filename, save_init=True):
        super().save(filename, save_init=save_init)
        save_dict(filename + FILENAME_SUFFIX, FeatureParameters.copy(self.params, copy_init=save_init))

    def load(self, filename, order=None):
        super().load(filename, order)
        self.params = FeatureParameters.copy(load_dict(filename + FILENAME_SUFFIX), UnknownDict, order=order)
        self.node_dropout = 0
