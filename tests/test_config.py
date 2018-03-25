"""Testing code for the tupa.config module, unit-testing only."""

import pytest

from tupa.config import Config


@pytest.fixture
def config():
    c = Config("", "-m", "test")
    c.update({"no_node_labels": True, "evaluate": True, "minibatch_size": 50})
    c.update_hyperparams(shared={"layer_dim": 50})
    return c


def test_params(config):
    d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1,
         "lstm_layer_dim": 100, "lstm_layers": 1}
    config.update(d)
    for attr, value in d.items():
        assert getattr(config.args, attr) == value, attr


def test_hyperparams(config):
    assert config.hyperparams.shared.layer_dim == 50, "--hyperparams=shared=--layer-dim=50"
    d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1}
    config.update(d)
    config.update_hyperparams(shared={"lstm_layer_dim": 100, "lstm_layers": 1}, ucca={"word_dim": 300},
                              **{"ucca.en": {"dep_dim": 1}})
    assert config.hyperparams.shared.lstm_layer_dim == 100, "shared --lstm-layer-dim=100"
    assert config.hyperparams.shared.lstm_layers == 1, "shared --lstm-layers=1"
    assert config.hyperparams.shared.minibatch_size == 50, "--minibatch-size=50"
    assert config.hyperparams.specific["ucca"].word_dim == 300, "ucca --word-dim=300"
    assert config.hyperparams.specific["ucca.en"].dep_dim == config.hyperparams.specific["ucca"]["en"].dep_dim == 1, \
        "ucca.en --dep-dim=1"
    assert config.hyperparams.specific["ucca"].minibatch_size == 50, "--minibatch-size=50"
    for attr, value in d.items():
        assert getattr(config.hyperparams.shared, attr) == value, attr


def test_boolean_params(config):
    assert config.args.evaluate
    assert not config.args.verify
