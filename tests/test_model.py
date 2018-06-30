"""Testing code for the tupa.model module, unit-testing only."""

import os
import pytest

from tupa.config import CLASSIFIERS
from tupa.model import Model, ClassifierProperty, NODE_LABEL_KEY
from tupa.states.state import State
from .conftest import remove_existing, weight_decay, assert_all_params_equal


def parse(formats, model, passage, train):
    for axis in formats:
        axes = (axis,) + ((NODE_LABEL_KEY,) if axis == "amr" else ())
        model.config.set_format(axis)
        model.init_model(axis)
        state = State(passage)
        if ClassifierProperty.require_init_features in model.classifier_properties:
            model.init_features(state, train=train)
        features = model.feature_extractor.extract_features(state)
        for a in axes:
            pred = model.classifier.score(features, axis=a).argmax()
            if train:
                model.classifier.update(features, axis=a, pred=pred, true=[0])
        model.classifier.finished_step(train=train)
        model.classifier.finished_item(train=train)


@pytest.mark.parametrize("iterations", (1, 2))
@pytest.mark.parametrize("model_type", CLASSIFIERS)
@pytest.mark.parametrize("omit_features", (None, "d"), ids=("", "omitd"))
def test_model(model_type, formats, test_passage, iterations, omit_features, config):
    filename = "_".join(filter(None, [os.path.join("test_files", "models", "test"), model_type, omit_features]+formats))
    remove_existing(filename)
    config.update(dict(classifier=model_type, copy_shared=None, omit_features=omit_features))
    finalized = model = Model(filename, config=config)
    for i in range(iterations):
        parse(formats, model, test_passage, train=True)
        finalized = model.finalize(finished_epoch=True)
        assert not getattr(finalized.feature_extractor, "node_dropout", 0), finalized.feature_extractor.node_dropout
        parse(formats, model, test_passage, train=False)
        finalized.save()
    loaded = Model(filename, config=config)
    loaded.load()
    assert not getattr(loaded.feature_extractor, "node_dropout", 0), loaded.feature_extractor.node_dropout
    for key, param in sorted(model.feature_extractor.params.items()):
        loaded_param = loaded.feature_extractor.params[key]
        assert param == loaded_param
    assert_all_params_equal(finalized.all_params(), loaded.all_params(), decay=weight_decay(model))
