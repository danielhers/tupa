"""Testing code for the tupa package, unit-testing only."""

from glob import glob
from itertools import combinations

import pytest
from ucca import convert, ioutil, textutil

from scheme.convert import FROM_FORMAT
from scheme.evaluate import Scores
from scheme.util.amr import WIKIFIER
from tupa.action import Actions
from tupa.config import Config, CLASSIFIERS
from tupa.model import Model, ClassifierProperty
from tupa.oracle import Oracle
from tupa.parse import Parser
from tupa.states.state import State

FORMATS = ("ucca", "amr", "conllu", "sdp")


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


def load_passages(passage_format=None):
    WIKIFIER.enabled = False
    return ioutil.read_files_and_dirs(
        glob("test_files/*." + {None: "*", "ucca": "xml"}.get(passage_format, passage_format)), converters=FROM_FORMAT)


def passage_id(passage):
    return passage.extra.get("format", "ucca")


@pytest.fixture
def config():
    config = Config("", "-m", "test")
    config.update({"verbose": 1, "timeout": 1,
                   "max_words_external": 100, "word_dim_external": 100, "word_dim": 10,
                   "max_words": 100, "max_node_labels": 20, "max_node_categories": 5,
                   "node_label_dim": 2, "node_category_dim": 2, "edge_label_dim": 2,
                   "tag_dim": 2, "dep_dim": 2, "optimizer": "sgd", "output_dim": 10,
                   "layer_dim": 15, "layers": 1, "lstm_layer_dim": 10, "lstm_layers": 1})
    config.update_hyperparams(shared={"lstm_layer_dim": 100, "lstm_layers": 1}, ucca={"word_dim": 300})
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


@pytest.mark.parametrize("setting", Settings.all(), ids=str)
@pytest.mark.parametrize("model_type", CLASSIFIERS)
@pytest.mark.parametrize("passage_format", FORMATS)
def test_parser(config, setting, model_type, passage_format):
    config.update(setting.dict())
    scores = []
    passages = load_passages(passage_format)
    for mode in "train", "load":
        print("-- %sing %s" % (mode, model_type))
        p = Parser(model_file="test_files/models/%s_%s%s" % (passage_format, model_type, setting.suffix()),
                   model_type=model_type)
        list(p.train(passages if mode == "train" else None, iterations=2))
        results = list(p.parse(passages, evaluate=True))
        score = Scores(tuple(zip(*results))[1])
        scores.append(score.average_f1())
        print("Converting to text and parsing...")
        text_results = list(p.parse([p3 for p1 in passages for p2 in convert.to_text(p1, sentences=False) for p3
                                     in convert.from_text(p2, p1.ID, extra_format=p1.extra.get("format"))]))
        assert len(results) == len(text_results)
        for t, (r, s) in zip(text_results, results):
            print("  %s F1=%.3f, text F1=%.3f" % (r.ID, s.average_f1(), p.evaluate_passage(t, r).average_f1()))
        assert not list(p.parse(()))  # parsing nothing returns nothing
        print()
    print("-- average labeled f1: %.3f, %.3f\n" % tuple(scores))
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


@pytest.mark.parametrize("model_type", CLASSIFIERS)
@pytest.mark.parametrize("passage", load_passages(), ids=passage_id)
def test_model(model_type, passage):
    filename = "test_files/models/test_%s_%s" % (model_type, passage.ID)
    axis = "test"
    Config().set_format(axis)
    textutil.annotate(passage)
    model = Model(model_type, filename)
    model.init_model()
    state = State(passage)
    if ClassifierProperty.require_init_features in model.get_classifier_properties():
        model.init_features(state, (axis,), train=True)
    features = model.feature_extractor.extract_features(state)
    pred = model.classifier.score(features, axis).argmax()
    model.classifier.update(features, axis, pred=pred, true=[0])
    model.finalize(finished_epoch=True).save()
    loaded = Model(model_type, filename)
    loaded.load(finalized=False)
    for suffix, param in sorted(model.feature_extractor.params.items()):
        loaded_param = loaded.feature_extractor.params[suffix]
        assert param == loaded_param
