"""Testing code for the tupa.features package, unit-testing only."""

from operator import itemgetter

import pytest

from tupa.action import Actions
from tupa.features.dense_features import DenseFeatureExtractor
from tupa.features.sparse_features import SparseFeatureExtractor
from tupa.model import Model
from tupa.oracle import Oracle
from tupa.states.state import State
from .conftest import load_passages, passage_id, config

PARAMS = {p.name: p.create_from_config() for p in Model(None, config=config()).param_defs}

FEATURE_EXTRACTORS = {
    "sparse": SparseFeatureExtractor(),
    "dense": DenseFeatureExtractor(PARAMS, indexed=False),
    "dense-indexed": DenseFeatureExtractor(PARAMS, indexed=True),
}


@pytest.mark.parametrize("feature_extractor_item", FEATURE_EXTRACTORS.items(), ids=itemgetter(0))
@pytest.mark.parametrize("passage", load_passages(), ids=passage_id)
def test_features(config, feature_extractor_item, passage, write_features):
    feature_extractor_id, feature_extractor = feature_extractor_item
    config.set_format(passage.extra.get("format"))
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
    compare_file = "test_files/features/%s-%s.txt" % (passage.ID, feature_extractor_id)
    if write_features:
        with open(compare_file, "w") as f:
            f.writelines(features)
    with open(compare_file) as f:
        assert f.readlines() == features, compare_file
