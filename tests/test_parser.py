"""Testing code for the parsing package, unit-testing only."""

import unittest

from parsing.config import Config
from parsing.oracle import Oracle
from parsing.parse import Parser
from parsing.state.state import State
from ucca import convert
from ucca.tests.test_ucca import TestUtil


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParserTests, self).__init__(*args, **kwargs)
        Config("", "-m", "test")
        self.passage = convert.from_standard(TestUtil.load_xml('test_files/standard3.xml'))

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
        with open('test_files/standard3.oracle_actions.txt') as f:
            self.assertSequenceEqual(actions_taken, f.readlines())

    def test_parser_sparse(self):
        passages = [self.passage]
        parsed = ParserUtil.train_test(passages, model_type="sparse")
        self.assertSequenceEqual(parsed, passages)

    def test_parser_dense(self):
        passages = [self.passage]
        parsed = ParserUtil.train_test(passages, model_type="dense")
        self.assertSequenceEqual(parsed, passages)

    def test_parser_nn(self):
        passages = [self.passage]
        parsed = ParserUtil.train_test(passages, model_type="nn")
        self.assertSequenceEqual(parsed, passages)


class ParserUtil:
    @staticmethod
    def train_test(passages, *args, **kwargs):
        p = Parser(*args, **kwargs)
        p.train(passages)
        _, parsed = zip(*p.parse(passages))
        return parsed
