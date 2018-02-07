"""Testing code for the tupa package, unit-testing only."""

import pytest
from numpy.testing import assert_allclose
from ucca import convert

from scheme.evaluate import Scores
from tupa.config import CLASSIFIERS, BIRNN
from tupa.parse import Parser
from .conftest import FORMATS, remove_existing, load_passages, weight_decay, assert_all_params_equal


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


@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_parser(config, model_type, formats, default_setting, text=True):
    filename = "test_files/models/%s_%s%s" % ("_".join(formats), model_type, default_setting.suffix())
    remove_existing(filename)
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
    remove_existing(filename)
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
        remove_existing(filename)
        list(Parser(model_files=filename, config=config).train(passages, dev=passages, iterations=2))
    list(Parser(model_files=filenames, config=config).parse(passages, evaluate=True))


@pytest.mark.parametrize("model_type", (BIRNN,))
def test_empty_features(empty_features_config, model_type):
    filename = "test_files/models/%s_%s_empty_features" % (FORMATS[0], model_type)
    remove_existing(filename)
    empty_features_config.update(dict(classifier=model_type))
    passages = load_passages(FORMATS[0])
    p = Parser(model_files=filename, config=empty_features_config)
    list(p.train(passages, dev=passages, test=True, iterations=2))
    list(p.parse(passages, evaluate=True))


# @pytest.mark.parametrize("model_type", (BIRNN,))
# def test_change_hyperparams_load(empty_features_config, model_type):
#     filename = "test_files/models/%s_%s_change_hyperparams" % (FORMATS[0], model_type)
#     remove_existing(filename)
#     c = empty_features_config.copy()
#     c.update(dict(classifier=model_type))
#     passages = load_passages(FORMATS[0])
#     for _ in range(2):
#         p = Parser(model_files=filename, config=c)
#         list(p.train(passages, dev=passages, test=True, iterations=2))
#         list(p.parse(passages, evaluate=True))
#         c.update({"ner_dim": 1, "action_dim": 1, "word_dim_external": 1, "word_dim": 1, "node_label_dim": 1,
#                   "node_category_dim": 1, "edge_label_dim": 1, "tag_dim": 1, "dep_dim": 1})
