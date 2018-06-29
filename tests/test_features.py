"""Testing code for the tupa.features package, unit-testing only."""
from collections import OrderedDict

import os
import pytest
from ucca import textutil

from tupa.action import Actions
from tupa.features.dense_features import DenseFeatureExtractor
from tupa.features.sparse_features import SparseFeatureExtractor
from tupa.model import Model
from tupa.oracle import Oracle
from tupa.states.state import State
from .conftest import passage_files, load_passage, basename

SPARSE = "sparse"
DENSE = "dense"
VOCAB = os.path.join("test_files", "vocab", "en_core_web_lg.csv")
WORD_VECTORS = os.path.join("test_files", "vocab", "wiki.en.vec")
OMITTED = "d"


class FeatureExtractorCreator:
    def __init__(self, name, indexed=False, annotated=False, vocab=None, wordvectors=None, omit=None):
        self.name = name
        self.indexed = indexed
        self.annotated = annotated
        self.vocab = vocab
        self.id = vocab == "-"
        self.wordvectors = wordvectors
        self.omit = omit

    def __str__(self):
        return "-".join([self.name] + [attr for attr in ("indexed", "annotated", "vocab", "id", "wordvectors", "omit")
                                       if getattr(self, attr)])

    def __call__(self, config):
        config.args.vocab = self.vocab
        config.args.word_vectors = self.wordvectors
        config.args.omit_features = self.omit
        return SparseFeatureExtractor(omit_features=self.omit) if self.name == SPARSE else DenseFeatureExtractor(
            OrderedDict((p.name, p.create_from_config()) for p in Model(None, config=config).param_defs()),
            indexed=self.indexed, node_dropout=0, omit_features=self.omit)


def feature_extractors(*args, **kwargs):
    return [FeatureExtractorCreator(SPARSE, *args, **kwargs), FeatureExtractorCreator(DENSE, *args, **kwargs),
            FeatureExtractorCreator(DENSE, *args, indexed=True, **kwargs)]


def extract_features(feature_extractor, state, features):
    values = feature_extractor.extract_features(state)
    if feature_extractor.params:
        for key, vs in values.items():
            assert len(vs) == feature_extractor.params[key].num, key
    features.append(values)


def _test_features(config, feature_extractor_creator, filename, write_features):
    feature_extractor = feature_extractor_creator(config)
    passage = load_passage(filename, annotate=feature_extractor_creator.annotated)
    textutil.annotate(passage, as_array=True, vocab=config.vocab())
    config.set_format(passage.extra.get("format") or "ucca")
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    for key, param in feature_extractor.params.items():
        if not param.numeric:
            param.dropout = 0
            feature_extractor.init_param(key)
    features = [feature_extractor.init_features(state)]
    while True:
        extract_features(feature_extractor, state, features)
        action = min(oracle.get_actions(state, actions).values(), key=str)
        state.transition(action)
        if state.need_label:
            extract_features(feature_extractor, state, features)
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
        if state.finished:
            break
    features = ["%s %s\n" % i for f in features if f for i in (sorted(f.items()) + [("", "")])]
    compare_file = os.path.join("test_files", "features", "-".join((basename(filename), str(feature_extractor_creator)))
                                + ".txt")
    if write_features:
        with open(compare_file, "w", encoding="utf-8") as f:
            f.writelines(features)
    with open(compare_file, encoding="utf-8") as f:
        assert f.readlines() == features, compare_file


@pytest.mark.parametrize("feature_extractor_creator",
                         [f for v in (None, "-", VOCAB) for w in (None, WORD_VECTORS) for o in (None, OMITTED)
                          for f in feature_extractors(vocab=v, wordvectors=w, omit=o)], ids=str)
@pytest.mark.parametrize("filename", passage_files(), ids=basename)
def test_features(config, feature_extractor_creator, filename, write_features):
    _test_features(config, feature_extractor_creator, filename, write_features)


@pytest.mark.parametrize("feature_extractor_creator",
                         [f for v in ("-", VOCAB) for w in (None, WORD_VECTORS) for o in (None, OMITTED)
                          for f in feature_extractors(annotated=True, vocab=v, wordvectors=w, omit=o)], ids=str)
@pytest.mark.parametrize("filename", passage_files("conllu"), ids=basename)
def test_features_conllu(config, feature_extractor_creator, filename, write_features):
    _test_features(config, feature_extractor_creator, filename, write_features)


@pytest.mark.parametrize("feature_extractor_creator", [f for o in (None, OMITTED)
                                                       for f in feature_extractors(omit=o)[:-1]], ids=str)
def test_feature_templates(config, feature_extractor_creator, write_features):
    config.set_format("amr")
    feature_extractor = feature_extractor_creator(config)
    features = ["%s\n" % i for i in feature_extractor.all_features()]
    compare_file = os.path.join("test_files", "features", "templates-%s.txt" % str(feature_extractor_creator))
    if write_features:
        with open(compare_file, "w") as f:
            f.writelines(features)
    with open(compare_file) as f:
        assert f.readlines() == features, compare_file
