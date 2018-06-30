from collections import OrderedDict

from .feature_extractor import FeatureExtractor, calc
from .feature_params import FeatureParameters, NumericFeatureParameters
from ..model_util import UNKNOWN_VALUE, MISSING_VALUE, UnknownDict, save_dict, load_dict

FEATURE_TEMPLATES = (
    "s0s1xd" "s1s0x" "s0b0xd" "b0s0x"  # specific edges
    "s0wmtudencpT#^$xhqyPCIEMN"
    "s1wmtudencT#^$xhyN"
    "s2wmtudencT#^$xhy"
    "s3wmtudencT#^$xhyN"
    "b0wmtudncT#^$hPCIEMN"
    "b1wmtudncT#^$"
    "b2wmtudncT#^$"
    "b3wmtudncT#^$"
    "s0lwmenc#^$"
    "s0rwmenc#^$"
    "s1lwmenc#^$"
    "s1rwmenc#^$"
    "s0llwmen#^$"
    "s0lrwmen#^$"
    "s0rlwmen#^$"
    "s0rrwmen#^$"
    "s1llwmen#^$"
    "s1lrwmen#^$"
    "s1rlwmen#^$"
    "s1rrwmen#^$"
    "s0Lwmen#^$"
    "s0Rwmen#^$"
    "s1Lwmen#^$"
    "s1Rwmen#^$"
    "b0Lwmen#^$"
    "b0Rwmen#^$"
    "s0b0e" "b0s0e"  # specific edges
    "a0eAa1eA",  # past actions
)
INDEXED = "wmtudT"  # words, lemmas, fine POS tags, coarse/universal POS tags, dep rels, entity type
DEFAULT = ()  # intermediate value for missing features
FILENAME_SUFFIX = ".enum"


class DenseFeatureExtractor(FeatureExtractor):
    """
    Extracts features from the parser state for classification. To be used with a NeuralNetwork classifier.
    """
    def __init__(self, params, indexed, hierarchical=False, node_dropout=0, init_params=True, omit_features=None):
        super().__init__(feature_templates=FEATURE_TEMPLATES, omit_features=omit_features)
        self.indexed = indexed
        self.hierarchical = hierarchical
        self.node_dropout = node_dropout
        if init_params:
            self.params = OrderedDict((k, p) for k, p in [(NumericFeatureParameters.SUFFIX, NumericFeatureParameters(1))
                                                          ] + list(params.items()))
            for param in self.params.values():
                self.update_param_indexed(param)
            num_values = self.num_values()
            for key, param in self.params.items():
                param.num = num_values[key]
                param.node_dropout = self.node_dropout
        else:
            self.params = params
    
    def init_param(self, key):
        param = self.params[key]
        self.update_param_indexed(param)
        param.num = self.num_values()[key]

    def num_values(self):
        return {k: len(v) for k, v in self.param_values(all_params=True).items()}

    def update_param_indexed(self, param):
        param.indexed = self.indexed and param.prop in INDEXED

    @property
    def feature_template(self):
        return self.feature_templates[0]

    def init_features(self, state):
        features = OrderedDict()
        for key, param in self.params.items():
            if param.indexed and param.enabled:
                values = [calc(n, state, param.prop) for n in state.terminals]
                param.init_data()
                features[key] = [param.data[v] for v in values]
        return features

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :return dict of feature name -> list of numeric values
        """
        features = OrderedDict()
        for key, values in self.param_values(state).items():
            param = self.params[key]
            param.init_data()  # Replace categorical values with their values in data dict:
            features[key] = [(UNKNOWN_VALUE if v == DEFAULT else v) if param.numeric else
                             (MISSING_VALUE if v == DEFAULT else (v if param.indexed else param.data[v]))
                             for v in values]
        return features

    def param_values(self, state=None, all_params=False):
        indexed = []
        by_key = OrderedDict()
        by_prop = OrderedDict()
        for key, param in self.params.items():
            if param.enabled and param.dim or all_params:
                if param.indexed:
                    if param.copy_from:
                        copy_from = self.params.get(param.copy_from)
                        if copy_from and copy_from.enabled and copy_from.dim and not all_params:
                            continue
                    if param.prop not in indexed:
                        indexed.append(param.prop)  # Only need one copy of indices
                by_key[key] = by_prop.setdefault(
                    NumericFeatureParameters.SUFFIX if param.numeric else param.prop,
                    ([state.node_ratio()] if state else [1] if all_params else []) if param.numeric else [])
        for e, prop, value in self.feature_template.extract(state, DEFAULT, "".join(indexed), as_tuples=True,
                                                            node_dropout=self.node_dropout,
                                                            hierarchical=self.hierarchical):
            vs = by_prop.get(NumericFeatureParameters.SUFFIX if e.is_numeric(prop) else prop)
            if vs is not None:
                vs.append(value if state else (e, prop))
        return by_key

    def all_features(self):
        return ["".join(self.join_props(vs)) for _, vs in sorted(self.param_values().items(), key=lambda x: x[0])]

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
        return type(self)(FeatureParameters.copy(self.params, UnknownDict), self.indexed, init_params=False,
                          omit_features=self.omit_features)

    def unfinalize(self):
        """Undo finalize(): replace each feature parameter's data dict with a DropoutDict again, to keep training"""
        for param in self.params.values():
            param.unfinalize()
            self.node_dropout = param.node_dropout

    def save(self, filename, save_init=True):  # TODO Save to JSON instead of pickle, with data as list (not dict)
        super().save(filename, save_init=save_init)
        save_dict(filename + FILENAME_SUFFIX, FeatureParameters.copy(self.params, copy_init=save_init))

    def load(self, filename, order=None):
        super().load(filename, order)
        self.params = FeatureParameters.copy(load_dict(filename + FILENAME_SUFFIX), UnknownDict, order=order)
        self.node_dropout = 0
