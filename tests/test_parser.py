"""Testing code for the parsing package, unit-testing only."""

import unittest

from parsing.config import Config
from parsing.oracle import Oracle
from parsing.parse import Parser
from parsing.state import State
from ucca import convert
from ucca.tests.test_ucca import TestUtil


class ParserTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ParserTests, self).__init__(*args, **kwargs)
        Config("", "-m", "test")

    def test_oracle(self):
        passage = convert.from_standard(TestUtil.load_xml('test_files/standard3.xml'))
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
        with open('test_files/standard3.oracle_actions.txt') as f:
            self.assertSequenceEqual(actions_taken, f.readlines())

    def test_parser(self):
        passages = [convert.from_standard(TestUtil.load_xml('test_files/standard3.xml'))]
        p = Parser()
        p.train(passages)
        _, parsed = zip(*p.parse(passages))
        self.assertSequenceEqual(parsed, passages)
