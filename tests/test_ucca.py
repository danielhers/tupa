"""Testing code for the ucca format, unit-testing only."""

import unittest
from glob import glob

from ucca.evaluation import evaluate
from ucca.ioutil import read_files_and_dirs


class EvaluationTests(unittest.TestCase):
    """Tests the evaluation module functions and classes."""

    def test_evaluate(self):
        """Test that comparing a passage against itself returns perfect F1"""
        for ref in read_test_passages():
            scores = evaluate(ref, ref)
            self.assertAlmostEqual(scores.average_f1(), 1)


def read_test_passages():
    yield from read_files_and_dirs(glob("test_files/*.xml"))
