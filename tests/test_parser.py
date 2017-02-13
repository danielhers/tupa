"""Testing code for the tupa package, unit-testing only."""

import unittest

from states.state import State
from tupa.config import Config, SPARSE_PERCEPTRON, DENSE_PERCEPTRON, MLP_NN, BILSTM_NN
from tupa.oracle import Oracle
from tupa.parse import Parser
from ucca import convert, evaluation, ioutil

NUM_PASSAGES = 2


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParserTests, self).__init__(*args, **kwargs)
        Config("", "-m", "test", "--linkage", "--implicit", "--no-constraints",
               "--max-words-external=100", "--word-dim-external=100", "--optimizer=sgd",
               "--layer-dim=100", "--layers=1", "--lstm-layer-dim=100", "--lstm-layers=1")

    @staticmethod
    def load_passages():
        passages = []
        for _ in range(NUM_PASSAGES):
            passages += ioutil.read_files_and_dirs(("ucca/test_files/standard3.xml",))
        return passages

    def test_oracle(self):
        for passage in self.load_passages():
            oracle = Oracle(passage)
            state = State(passage)
            actions_taken = []
            while True:
                actions = oracle.get_actions(state)
                action = next(iter(actions))
                state.transition(action)
                actions_taken.append("%s\n" % action)
                if state.finished:
                    break
            with open("test_files/standard3.oracle_actions.txt") as f:
                self.assertSequenceEqual(actions_taken, f.readlines())

    def test_parser_sparse(self):
        self.train_test(SPARSE_PERCEPTRON)

    def test_parser_dense(self):
        self.train_test(DENSE_PERCEPTRON)

    def test_parser_mlp(self):
        self.train_test(MLP_NN)

    def test_parser_bilstm(self):
        self.train_test(BILSTM_NN)

    def train_test(self, model_type, compare=True):
        scores = []
        p = None
        for mode in "train", "load":
            print("-- %sing %s" % (mode, model_type))
            p = Parser(model_file="test_files/models/%s" % model_type, model_type=model_type)
            p.train(self.load_passages() if mode == "train" else None, iterations=200)
            score = evaluation.Scores.aggregate([s for _, s in p.parse(self.load_passages(), evaluate=True)])
            scores.append(score.average_f1())
            print()
        print("-- average labeled f1: %.3f, %.3f\n" % tuple(scores))
        if compare:
            self.assertAlmostEqual(*scores)
        p.parse(convert.to_text(self.load_passages()[0]))
        self.assertFalse(list(p.parse(())))  # parsing nothing returns nothing
