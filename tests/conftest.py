from itertools import combinations

import numpy as np
import os
import pytest
from functools import partial
from glob import glob
from numpy.testing import assert_allclose, assert_array_equal
from semstr.convert import FROM_FORMAT
from semstr.util.amr import WIKIFIER
from ucca import ioutil

from tupa.config import Config

FORMATS = ("ucca", "amr", "conllu", "sdp")


def pytest_addoption(parser):
    parser.addoption("--write-oracle-actions", action="store_true", help="overwrite oracle action expected results")
    parser.addoption("--write-features", action="store_true", help="overwrite feature expected results")
    parser.addoption("--multitask", action="store_true", help="test multitask parsing")


@pytest.fixture
def write_oracle_actions(request):
    return request.config.getoption("--write-oracle-actions")


@pytest.fixture
def write_features(request):
    return request.config.getoption("--write-features")


def pytest_generate_tests(metafunc):
    if "formats" in metafunc.fixturenames:
        formats = [[f] for f in FORMATS]
        if metafunc.config.getoption("--multitask"):
            formats += [[FORMATS[0], f] for f in FORMATS[1:]]
        metafunc.parametrize("formats", formats, ids="-".join)


@pytest.fixture
def config():
    c = Config("", "-m", "test")
    c.update({"verbose": 2, "timeout": 1, "embedding_layer_dim": 1, "ner_dim": 1, "action_dim": 1, "lemma_dim": 1,
              "max_words_external": 3, "word_dim_external": 1, "word_dim": 1, "max_words": 3, "max_lemmas": 3,
              "max_tags": 3, "max_pos": 3, "max_deps": 3, "max_edge_labels": 3, "max_puncts": 3, "max_action_types": 3,
              "max_ner_types": 3, "edge_label_dim": 1, "tag_dim": 1, "pos_dim": 1, "dep_dim": 1, "optimizer": "sgd",
              "output_dim": 1, "layer_dim": 2, "layers": 3, "lstm_layer_dim": 2, "lstm_layers": 3,
              "max_action_ratio": 10, "update_word_vectors": False, "copy_shared": None})
    c.update_hyperparams(shared={"lstm_layer_dim": 2, "lstm_layers": 1}, ucca={"word_dim": 2},
                         amr={"max_node_labels": 3, "max_node_categories": 3,
                              "node_label_dim": 1, "node_category_dim": 1})
    return c


@pytest.fixture
def empty_features_config():
    c = config()
    c.update({"ner_dim": 0, "action_dim": 0, "word_dim_external": 0, "word_dim": 0, "lemma_dim": 0, "node_label_dim": 0,
              "node_category_dim": 0, "edge_label_dim": 0, "tag_dim": 0, "pos_dim": 0, "dep_dim": 0})
    return c


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


def passage_files(*formats):
    return [f for fo in formats or ["*"] for f in glob("test_files/*." + ("xml" if fo == "ucca" else fo))
            if not f.endswith(".txt")]


def load_passage(filename, annotate=False):
    WIKIFIER.enabled = False
    converters = {k: partial(c, annotate=annotate) for k, c in FROM_FORMAT.items()}
    passages = ioutil.read_files_and_dirs(filename, converters=converters, attempts=1, delay=0)
    try:
        return next(iter(passages))
    except StopIteration:
        return passages


def basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


@pytest.fixture
def test_passage():
    return next(iter(ioutil.read_files_and_dirs(("test_files/120.xml",))))


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


def remove_existing(filename):
    for f in glob(filename + ".*"):
        os.remove(f)
