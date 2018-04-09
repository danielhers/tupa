"""Testing code for the amr format, unit-testing only."""

import unittest

from ucca.convert import split2sentences

from semstr.conversion.amr import from_amr, to_amr
from semstr.evaluation.amr import evaluate


class ConversionTests(unittest.TestCase):
    """Tests conversion module correctness and API."""

    def test_convert(self):
        """Test that converting an AMR to UCCA and back retains perfect Smatch F1"""
        for passage, ref, amr_id in read_test_amr():
            self.convert_and_evaluate(amr_id, passage, ref)

    def test_split(self):
        """Test that splitting a single-sentence AMR converted to UCCA returns the same AMR"""
        for passage, ref, amr_id in read_test_amr():
            sentences = split2sentences(passage)
            self.assertEqual(len(sentences), 1, "Should be one sentence: %s" % passage)
            sentence = sentences[0]
            self.convert_and_evaluate(amr_id, sentence, ref)

    def convert_and_evaluate(self, amr_id, passage, ref):
        converted = "\n".join(to_amr(passage, metadata=False))
        scores = evaluate(converted, ref, amr_id=amr_id)
        self.assertAlmostEqual(scores.average_f1(), 1, msg=converted)

    def test_compare(self):
        """Test that converting an AMR to UCCA gives the same passage"""
        passages = [list(read_test_amr()) for _ in range(2)]
        for (passage1, ref1, amr_id1), (passage2, ref2, amr_id2) in zip(*passages):
            self.assertTrue(passage1.equals(passage2), amr_id1)


class EvaluationTests(unittest.TestCase):
    """Tests the evaluation module functions and classes."""

    def test_evaluate(self):
        """Test that comparing an AMR against itself returns perfect Smatch F1"""
        for _, ref, amr_id in read_test_amr():
            scores = evaluate(ref, ref, amr_id=amr_id)
            self.assertAlmostEqual(scores.average_f1(), 1)


def read_test_amr():
    with open("test_files/LDC2014T12.amr") as f:
        yield from from_amr(f, return_original=True)
