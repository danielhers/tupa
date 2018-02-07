"""Testing code for the tupa.features package, unit-testing only."""

import pytest

from tupa.action import Actions
from tupa.features.dense_features import DenseFeatureExtractor
from tupa.features.sparse_features import SparseFeatureExtractor
from tupa.model import Model
from tupa.oracle import Oracle
from tupa.states.state import State
from .conftest import load_passages, passage_id


class FeatureExtractorCreator:
    def __init__(self, name, indexed=False):
        self.name = name
        self.indexed = indexed

    def __str__(self):
        return self.name + ("-indexed" if self.indexed else "")

    def __call__(self, config):
        if self.name == "sparse":
            return SparseFeatureExtractor()
        return DenseFeatureExtractor({p.name: p.create_from_config() for p in Model(None, config=config).param_defs},
                                     indexed=self.indexed)


FEATURE_EXTRACTORS = [
    FeatureExtractorCreator("sparse"), FeatureExtractorCreator("dense"), FeatureExtractorCreator("dense", indexed=True),
]


@pytest.mark.parametrize("feature_extractor_creator", FEATURE_EXTRACTORS, ids=str)
@pytest.mark.parametrize("passage", load_passages(), ids=passage_id)
def test_features(config, feature_extractor_creator, passage, write_features):
    config.set_format(passage.extra.get("format"))
    feature_extractor = feature_extractor_creator(config)
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    for param in feature_extractor.params.values():
        if not param.numeric:
            feature_extractor.init_param(param)
    features = [feature_extractor.init_features(state)]
    while True:
        features.append(feature_extractor.extract_features(state))
        action = min(oracle.get_actions(state, actions).values(), key=str)
        state.transition(action)
        if state.need_label:
            features.append(feature_extractor.extract_features(state))
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
        if state.finished:
            break
    features = ["%s %s\n" % i for f in features if f for i in (sorted(f.items()) + [("", "")])]
    compare_file = "test_files/features/%s-%s.txt" % (passage.ID, feature_extractor_creator.name)
    if write_features:
        with open(compare_file, "w") as f:
            f.writelines(features)
    with open(compare_file) as f:
        assert f.readlines() == features, compare_file


@pytest.mark.parametrize("feature_extractor_creator", FEATURE_EXTRACTORS[:-1], ids=str)
def test_feature_templates(config, feature_extractor_creator, write_features):
    feature_extractor = feature_extractor_creator(config)
    features = [feature_extractor.numeric_features_template if p.numeric else
                feature_extractor.non_numeric_feature_templates[p.effective_suffix] for _, p in
                sorted(feature_extractor.params.items())] or sorted(feature_extractor.feature_templates, key=str)
    features = ["%s\n" % i for i in features]
    compare_file = "test_files/features/templates-%s.txt" % feature_extractor_creator.name
    if write_features:
        with open(compare_file, "w") as f:
            f.writelines(features)
    with open(compare_file) as f:
        assert f.readlines() == features, compare_file
