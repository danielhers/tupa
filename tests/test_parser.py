"""Testing code for the tupa package, unit-testing only."""

import os
import unittest
from glob import glob

from ucca import convert, ioutil

from scheme.convert import FROM_FORMAT
from scheme.evaluate import Scores
from tupa.action import Actions
from tupa.config import Config, SPARSE, MLP_NN, BILSTM_NN, NOOP
from tupa.oracle import Oracle
from tupa.parse import Parser
from tupa.states.state import State

TOY_DATA = glob(os.environ.get("TOY_DATA", "test_files/*.xml"))
SETTINGS = ([], ["implicit"], ["linkage"], ["implicit", "linkage"])
NUM_PASSAGES = 2


def update_settings(settings):
    print("-- settings: " + ", ".join(settings))
    Config().update({s: s in settings for s in ("implicit", "linkage")})


def settings_suffix(settings):
    return "_".join([""] + settings)


def load_passages():
    passages = []
    for _ in range(NUM_PASSAGES):
        passages += ioutil.read_files_and_dirs(TOY_DATA, converters=FROM_FORMAT)
    return passages


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Config("", "-m", "test")
        Config().update({"max_words_external": 100, "word_dim_external": 100, "word_dim": 10,
                         "max_words": 100, "max_node_labels": 20, "max_node_categories": 5,
                         "tag_dim": 5, "dep_dim": 5, "optimizer": "sgd", "output_dim": 35,
                         "layer_dim": 30, "layers": 1, "lstm_layer_dim": 40, "lstm_layers": 1})
        Config().update_hyperparams(shared={"lstm_layer_dim": 100, "lstm_layers": 1}, ucca={"word_dim": 300})

    def test_oracle(self):
        self.maxDiff = None
        for settings in SETTINGS:
            update_settings(settings)
            for passage in load_passages():
                Config().set_format(passage.extra.get("format"))
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
                compare_file = "test_files/oracle_actions/%s%s.txt" % (passage.ID, settings_suffix(settings))
                # with open(compare_file, "w") as f:
                #     f.writelines(actions_taken)
                with open(compare_file) as f:
                    self.assertSequenceEqual(f.readlines(), actions_taken)

    def test_parser_sparse(self):
        self.train_test(SPARSE)

    def test_parser_mlp(self):
        self.train_test(MLP_NN, compare=False)

    def test_parser_bilstm(self):
        self.train_test(BILSTM_NN, compare=False)

    def test_parser_noop(self):
        self.train_test(NOOP)

    def train_test(self, model_type, compare=True):
        for settings in SETTINGS:
            update_settings(settings)
            scores = []
            p = None
            for mode in "train", "load":
                print("-- %sing %s" % (mode, model_type))
                model_filename = model_type + settings_suffix(settings)
                p = Parser(model_file="test_files/models/%s" % model_filename, model_type=model_type)
                list(p.train(load_passages() if mode == "train" else None, iterations=10))
                score = Scores([s for _, s in p.parse(load_passages(), evaluate=True)])
                scores.append(score.average_f1())
                print()
            print("-- average labeled f1: %.3f, %.3f\n" % tuple(scores))
            if compare:
                self.assertAlmostEqual(*scores, places=2)
            p.parse(convert.to_text(load_passages()[0]))
            self.assertFalse(list(p.parse(())))  # parsing nothing returns nothing


class ConfigTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Config("", "-m", "test")
        Config().update({"no_node_labels": True, "evaluate": True, "minibatch_size": 50})
        Config().update_hyperparams(shared={"layer_dim": 50})

    def test_params(self):
        d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1,
             "lstm_layer_dim": 100, "lstm_layers": 1}
        Config().update(d)
        for attr, value in d.items():
            self.assertEqual(getattr(Config().args, attr), value, attr)

    def test_hyperparams(self):
        self.assertEqual(Config().hyperparams.shared.layer_dim, 50, "--hyperparams=shared=--layer-dim=50")
        d = {"max_words_external": 100, "word_dim_external": 100, "optimizer": "sgd", "layer_dim": 100, "layers": 1}
        Config().update(d)
        Config().update_hyperparams(shared={"lstm_layer_dim": 100, "lstm_layers": 1}, ucca={"word_dim": 300})
        self.assertEqual(Config().hyperparams.shared.lstm_layer_dim, 100, "shared --lstm-layer-dim=100")
        self.assertEqual(Config().hyperparams.shared.lstm_layers, 1, "shared --lstm-layers=1")
        self.assertEqual(Config().hyperparams.shared.minibatch_size, 50, "--minibatch-size=50")
        self.assertEqual(Config().hyperparams.specific["ucca"].word_dim, 300, "ucca --word-dim=300")
        self.assertEqual(Config().hyperparams.specific["ucca"].minibatch_size, 50, "--minibatch-size=50")
        for attr, value in d.items():
            self.assertEqual(getattr(Config().hyperparams.shared, attr), value, attr)

    def test_boolean_params(self):
        self.assertTrue(Config().args.evaluate)
        self.assertFalse(Config().args.verify)


if __name__ == "__main__":
    unittest.main()
