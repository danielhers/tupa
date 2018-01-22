"""Testing code for the tupa package, unit-testing only."""

import os
from glob import glob
from itertools import combinations

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ucca import convert, ioutil

from scheme.convert import FROM_FORMAT
from scheme.evaluate import Scores
from scheme.util.amr import WIKIFIER
from tupa.action import Actions
from tupa.config import Config, CLASSIFIERS
from tupa.model import Model, ClassifierProperty, NODE_LABEL_KEY
from tupa.oracle import Oracle
from tupa.parse import Parser
from tupa.states.state import State


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


def load_passages(*formats):
    WIKIFIER.enabled = False
    files = [f for fo in formats or ["*"] for f in glob("test_files/*." + ("xml" if fo == "ucca" else fo))]
    return ioutil.read_files_and_dirs(files, converters=FROM_FORMAT)


def passage_id(passage):
    return passage.extra.get("format", "ucca")


@pytest.fixture
def config():
    config = Config("", "-m", "test")
    config.update({"verbose": 2, "timeout": 1, "embedding_layer_dim": 1, "ner_dim": 1, "action_dim": 1,
                   "max_words_external": 3, "word_dim_external": 1, "word_dim": 1,
                   "max_words": 3, "max_node_labels": 3, "max_node_categories": 3,
                   "max_tags": 3, "max_deps": 3, "max_edge_labels": 3, "max_puncts": 3, "max_action_types": 3,
                   "max_ner_types": 3, "node_label_dim": 1, "node_category_dim": 1, "edge_label_dim": 1,
                   "tag_dim": 1, "dep_dim": 1, "optimizer": "sgd", "output_dim": 1,
                   "layer_dim": 1, "layers": 1, "lstm_layer_dim": 2, "lstm_layers": 1, "max_action_ratio": 10})
    # "use_gold_node_labels": True})
    config.update_hyperparams(shared={"lstm_layer_dim": 2, "lstm_layers": 1}, ucca={"word_dim": 2})
    return config


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


@pytest.fixture
def default_setting():
    return Settings()


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
        p = Parser(model_file=filename, model_type=model_type)
        list(p.train(passages if mode == "train" else None, dev=passages, test=True, iterations=2))
        assert p.model.is_finalized, "Model should be finalized after %sing" % mode
        params.append(p.model.get_all_params())
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


@pytest.fixture
def test_config():
    config = Config("", "-m", "test")
    config.update({"no_node_labels": True, "evaluate": True, "minibatch_size": 50})
    config.update_hyperparams(shared={"layer_dim": 50})
    return config


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


@pytest.fixture
def test_passage():
    return next(iter(ioutil.read_files_and_dirs(("test_files/120.xml",))))


@pytest.mark.parametrize("iterations", (1, 2))
@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_model(model_type, formats, test_passage, iterations):
    filename = "test_files/models/test_%s_%s" % (model_type, "_".join(formats))
    for f in glob(filename + ".*"):
        os.remove(f)
    finalized = model = Model(model_type, filename)
    for i in range(iterations):
        parse(formats, model, test_passage, train=True)
        finalized = model.finalize(finished_epoch=True)
        parse(formats, model, test_passage, train=False)
        finalized.save()
    loaded = Model(model_type, filename)
    loaded.load()
    for key, param in sorted(model.feature_extractor.params.items()):
        loaded_param = loaded.feature_extractor.params[key]
        assert param == loaded_param
    assert_all_params_equal(finalized.get_all_params(), loaded.get_all_params())


def assert_all_params_equal(*params):
    for key, param in sorted(params[0].items()):
        for p in params[1:]:
            exception = None
            for f in (assert_array_almost_equal, assert_array_equal):
                try:
                    f(param, p[key], key, verbose=True)
                    exception = None
                except TypeError as e:
                    exception = e
            if exception:
                raise exception


def parse(formats, model, passage, train):
    for axis in formats:
        axes = (axis,) + ((NODE_LABEL_KEY,) if axis == "amr" else ())
        Config().set_format(axis)
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
