"""Testing code for the parsing package, unit-testing only."""

import unittest

from parsing import config
from parsing.config import Config
from parsing.oracle import Oracle
from parsing.parse import Parser
from states.state import State
from ucca import convert, evaluation
from ucca.tests.test_ucca import TestUtil


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParserTests, self).__init__(*args, **kwargs)
        Config("", "-m", "test")
        self.passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))

    def test_oracle(self):
        oracle = Oracle(self.passage)
        state = State(self.passage)
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
        self.train_test(config.SPARSE_PERCEPTRON)

    def test_parser_dense(self):
        self.train_test(config.DENSE_PERCEPTRON)

    def test_parser_nn(self):
        self.train_test(config.MLP)

    def train_test(self, model_type, compare=True):
        passages = [self.passage]
        scores = []
        for mode in "train", "load":
            print("-- %sing %s" % (mode, model_type))
            p = Parser(model_file="test_files/%s" % model_type, model_type=model_type)
            p.train(passages if mode == "train" else None)
            guess, ref = zip(*list(p.parse(passages)))
            print()
            self.assertSequenceEqual(ref, passages)
            score = evaluation.Scores.aggregate([evaluation.evaluate(
                g, r, verbose=False, units=False, errors=False)
                                                 for g, r in zip(guess, ref)])
            scores.append(score.average_f1())
        print("-- average labeled f1: %.3f, %.3f\n" % tuple(scores))
        if compare:
            self.assertEqual(*scores)
