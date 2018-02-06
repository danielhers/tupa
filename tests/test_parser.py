"""Testing code for the tupa package, unit-testing only."""

import os
from glob import glob
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from ucca import convert, ioutil

from scheme.convert import FROM_FORMAT
from scheme.evaluate import Scores
from scheme.util.amr import WIKIFIER
from tupa.action import Actions
from tupa.config import Config, CLASSIFIERS, BIRNN
from tupa.model import Model, ClassifierProperty, NODE_LABEL_KEY
from tupa.oracle import Oracle
from tupa.parse import Parser
from tupa.states.state import State
from .conftest import FORMATS


# noinspection PyUnresolvedReferences
class Settings:
    SETTINGS = ("implicit", "linkage", "unlabeled")
    VALUES = {"unlabeled": (None, [])}
    INCOMPATIBLE = (("linkage", "unlabeled"),)

    def __init__(self, *args):
        for attr in self.SETTINGS:
            setattr(self, attr, attr in args)

    @classmethod
    def all(cls):
        return [Settings(*c) for n in range(len(cls.SETTINGS) + 1) for c in combinations(cls.SETTINGS, n)
                if not any(all(s in c for s in i) for i in cls.INCOMPATIBLE)]

    def dict(self):
        return {attr: self.VALUES.get(attr, (False, True))[getattr(self, attr)] for attr in self.SETTINGS}

    def list(self):
        return [attr for attr in self.SETTINGS if getattr(self, attr)]

    def suffix(self):
        return "_".join([""] + self.list())

    def __str__(self):
        return "-".join(self.list()) or "default"


@pytest.fixture
def default_setting():
    return Settings()


def load_passages(*formats):
    WIKIFIER.enabled = False
    files = [f for fo in formats or ["*"] for f in glob("test_files/*." + ("xml" if fo == "ucca" else fo))]
    return ioutil.read_files_and_dirs(files, converters=FROM_FORMAT)


def passage_id(passage):
    return passage.extra.get("format", "ucca")


@pytest.fixture
def test_passage():
    return next(iter(ioutil.read_files_and_dirs(("test_files/120.xml",))))


@pytest.fixture
def config():
    c = Config("", "-m", "test")
    c.update({"verbose": 2, "timeout": 1, "embedding_layer_dim": 1, "ner_dim": 1, "action_dim": 1,
              "max_words_external": 3, "word_dim_external": 1, "word_dim": 1,
              "max_words": 3, "max_node_labels": 3, "max_node_categories": 3,
              "max_tags": 3, "max_deps": 3, "max_edge_labels": 3, "max_puncts": 3, "max_action_types": 3,
              "max_ner_types": 3, "node_label_dim": 1, "node_category_dim": 1, "edge_label_dim": 1,
              "tag_dim": 1, "dep_dim": 1, "optimizer": "sgd", "output_dim": 1,
              "layer_dim": 2, "layers": 3, "lstm_layer_dim": 2, "lstm_layers": 3, "max_action_ratio": 10,
              "update_word_vectors": False})
    # "use_gold_node_labels": True})
    c.update_hyperparams(shared={"lstm_layer_dim": 2, "lstm_layers": 1}, ucca={"word_dim": 2})
    return c


@pytest.fixture
def empty_features_config():
    c = config()
    c.update({"ner_dim": 0, "action_dim": 0, "word_dim_external": 0, "word_dim": 0, "node_label_dim": 0,
              "node_category_dim": 0, "edge_label_dim": 0, "tag_dim": 0, "dep_dim": 0})
    return c


@pytest.fixture
def test_config():
    c = Config("", "-m", "test")
    c.update({"no_node_labels": True, "evaluate": True, "minibatch_size": 50})
    c.update_hyperparams(shared={"layer_dim": 50})
    return c


def weight_decay(model):
    try:
        return np.float_power(1 - model.classifier.weight_decay, model.classifier.updates)
    except AttributeError:
        return 1


def assert_all_params_equal(*params, decay=1):
    for key, param in sorted(params[0].items()):
        for p in params[1:]:
            try:
                assert_allclose(decay * param, p[key], rtol=1e-6)
            except TypeError:
                assert_array_equal(param, p[key])


def parse(formats, model, passage, train):
    for axis in formats:
        axes = (axis,) + ((NODE_LABEL_KEY,) if axis == "amr" else ())
        model.config.set_format(axis)
        model.init_model()
        state = State(passage)
        if ClassifierProperty.require_init_features in model.get_classifier_properties():
            model.init_features(state, axes, train=train)
        features = model.feature_extractor.extract_features(state)
        for a in axes:
            pred = model.classifier.score(features, axis=a).argmax()
            if train:
                model.classifier.update(features, axis=a, pred=pred, true=[0])
        model.classifier.finished_step(train=train)
        model.classifier.finished_item(train=train)


@pytest.mark.parametrize("setting", Settings.all(), ids=str)
@pytest.mark.parametrize("passage", load_passages(), ids=passage_id)
def test_oracle(config, setting, passage, write_oracle_actions):
    config.update(setting.dict())
    config.set_format(passage.extra.get("format"))
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    actions_taken = []
    while True:
        action = min(oracle.get_actions(state, actions).values(), key=str)
        state.transition(action)
        s = str(action)
        if state.need_label:
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
            s += " " + str(label)
        actions_taken.append(s + "\n")
        if state.finished:
            break
    compare_file = "test_files/oracle_actions/%s%s.txt" % (passage.ID, setting.suffix())
    if write_oracle_actions:
        with open(compare_file, "w") as f:
            f.writelines(actions_taken)
    with open(compare_file) as f:
        assert f.readlines() == actions_taken, compare_file


@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_parser(config, model_type, formats, default_setting, text=True):
    filename = "test_files/models/%s_%s%s" % ("_".join(formats), model_type, default_setting.suffix())
    for f in glob(filename + ".*"):
        os.remove(f)
    config.update(default_setting.dict())
    scores = []
    params = []
    passages = load_passages(*formats)
    evaluate = ("amr" not in formats)
    for mode in "train", "load":
        print("-- %sing %s" % (mode, model_type))
        config.update(dict(classifier=model_type, copy_shared=None))
        p = Parser(model_files=filename, config=config)
        p.save_init = True
        list(p.train(passages if mode == "train" else None, dev=passages, test=True, iterations=2))
        assert p.model.is_finalized, "Model should be finalized after %sing" % mode
        all_params = p.model.get_all_params()
        params.append(all_params)
        param1, param2 = [d.get("W") for d in (all_params, p.model.feature_extractor.params)]
        if param1 is not None and param2 and param2.init is not None and not config.args.update_word_vectors:
            assert_allclose(param1, weight_decay(p.model) * param2.init, rtol=1e-6)
        text_results = results = list(p.parse(passages, evaluate=evaluate))
        if text:
            print("Converting to text and parsing...")
            text_results = list(p.parse([p3 for p1 in passages for p2 in convert.to_text(p1, sentences=False) for p3
                                         in convert.from_text(p2, p1.ID, extra_format=p1.extra.get("format"))]))
            assert len(results) == len(text_results)
        if evaluate:
            scores.append(Scores(tuple(zip(*results))[1]).average_f1())
            if text:
                for t, (r, s) in zip(text_results, results):
                    print("  %s F1=%.3f" % (r.ID, s.average_f1()))
        assert not list(p.parse(()))  # parsing nothing returns nothing
        print()
    assert_all_params_equal(*params)
    if evaluate:
        print("-- average f1: %.3f, %.3f\n" % tuple(scores))
        assert scores[0] == pytest.approx(scores[1], 0.1)


@pytest.mark.parametrize("model_type", (BIRNN,))
def test_copy_shared(config, model_type):
    filename = "test_files/models/%s_%s_copy_shared" % ("_".join(FORMATS), model_type)
    for f in glob(filename + ".*"):
        os.remove(f)
    config.update(dict(classifier=model_type, lstm_layers=0, copy_shared=[FORMATS[0]]))
    for formats in ((FORMATS[0],), FORMATS):
        p = Parser(model_files=filename, config=config)
        passages = load_passages(*formats)
        list(p.train(passages, dev=passages, test=True, iterations=2))
        list(p.parse(passages, evaluate=True))
        config.update_hyperparams(ucca={"lstm_layers": 1})


@pytest.mark.parametrize("model_type", (BIRNN,))
def test_ensemble(config, model_type):
    config.update(dict(classifier=model_type, lstm_layers=0))
    filenames = ["test_files/models/%s_%s_ensemble%d" % (FORMATS[0], model_type, i) for i in range(1, 3)]
    passages = load_passages(FORMATS[0])
    for i, filename in enumerate(filenames, start=1):
        config.update(dict(seed=i))
        for f in glob(filename + ".*"):
            os.remove(f)
        list(Parser(model_files=filename, config=config).train(passages, dev=passages, iterations=2))
    list(Parser(model_files=filenames, config=config).parse(passages, evaluate=True))


@pytest.mark.parametrize("model_type", (BIRNN,))
def test_empty_features(empty_features_config, model_type):
    filename = "test_files/models/%s_%s_empty_features" % (FORMATS[0], model_type)
    for f in glob(filename + ".*"):
        os.remove(f)
    empty_features_config.update(dict(classifier=model_type))
    passages = load_passages(FORMATS[0])
    p = Parser(model_files=filename, config=empty_features_config)
    list(p.train(passages, dev=passages, test=True, iterations=2))
    list(p.parse(passages, evaluate=True))


def test_params(test_config):
    d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1,
         "lstm_layer_dim": 100, "lstm_layers": 1}
    test_config.update(d)
    for attr, value in d.items():
        assert getattr(test_config.args, attr) == value, attr


def test_hyperparams(test_config):
    assert test_config.hyperparams.shared.layer_dim == 50, "--hyperparams=shared=--layer-dim=50"
    d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1}
    test_config.update(d)
    test_config.update_hyperparams(shared={"lstm_layer_dim": 100, "lstm_layers": 1}, ucca={"word_dim": 300})
    assert test_config.hyperparams.shared.lstm_layer_dim == 100, "shared --lstm-layer-dim=100"
    assert test_config.hyperparams.shared.lstm_layers == 1, "shared --lstm-layers=1"
    assert test_config.hyperparams.shared.minibatch_size == 50, "--minibatch-size=50"
    assert test_config.hyperparams.specific["ucca"].word_dim == 300, "ucca --word-dim=300"
    assert test_config.hyperparams.specific["ucca"].minibatch_size == 50, "--minibatch-size=50"
    for attr, value in d.items():
        assert getattr(test_config.hyperparams.shared, attr) == value, attr


def test_boolean_params(test_config):
    assert test_config.args.evaluate
    assert not test_config.args.verify


@pytest.mark.parametrize("iterations", (1, 2))
@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_model(model_type, formats, test_passage, iterations, config):
    filename = "test_files/models/test_%s_%s" % (model_type, "_".join(formats))
    for f in glob(filename + ".*"):
        os.remove(f)
    config.update(dict(classifier=model_type, copy_shared=None))
    finalized = model = Model(filename, config=config)
    for i in range(iterations):
        parse(formats, model, test_passage, train=True)
        finalized = model.finalize(finished_epoch=True)
        parse(formats, model, test_passage, train=False)
        finalized.save()
    loaded = Model(filename, config=config)
    loaded.load()
    for key, param in sorted(model.feature_extractor.params.items()):
        loaded_param = loaded.feature_extractor.params[key]
        assert param == loaded_param
    assert_all_params_equal(finalized.get_all_params(), loaded.get_all_params(), decay=weight_decay(model))
