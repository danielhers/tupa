"""Testing code for the parsing package, unit-testing only."""

import unittest

from parsing.config import Config, SPARSE_PERCEPTRON, DENSE_PERCEPTRON, FEEDFORWARD_NN
from parsing.oracle import Oracle
from parsing.parse import Parser
from states.state import State
from ucca import convert, evaluation
from ucca.tests.test_ucca import TestUtil


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParserTests, self).__init__(*args, **kwargs)
        Config("", "-m", "test", "--maxwordsexternal", "100",
               "--layerdim", "100", "--layers", "1", "--updatewordvectors")
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
        self.train_test(SPARSE_PERCEPTRON)

    def test_parser_dense(self):
        self.train_test(DENSE_PERCEPTRON)

    def test_parser_nn(self):
        self.train_test(FEEDFORWARD_NN)

    def train_test(self, model_type, compare=True):
        passages = [self.passage] * 2
        scores = []
        for mode in "train", "load":
            print("-- %sing %s" % (mode, model_type))
            p = Parser(model_file="test_files/%s" % model_type, model_type=model_type)
            p.train(passages if mode == "train" else None, iterations=2)
            score = evaluation.Scores.aggregate([s for _, s in p.parse(passages, evaluate=True)])
            scores.append(round(score.average_f1(), 1))
            print()
        print("-- average labeled f1: %.1f, %.1f\n" % tuple(scores))
        if compare:
            self.assertAlmostEqual(*scores)
        p.parse(convert.to_text(self.passage))
        self.assertFalse(list(p.parse(())))  # parsing nothing returns nothing
