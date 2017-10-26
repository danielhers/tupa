"""Testing code for the sdp format, unit-testing only."""

import unittest

from ucca.convert import split2sentences

from scheme.conversion.sdp import from_sdp, to_sdp
from scheme.evaluation.sdp import evaluate


class ConversionTests(unittest.TestCase):
    """Tests conversion module correctness and API."""

    def test_convert(self):
        """Test that converting an SDP graph to UCCA and back retains perfect Smatch F1"""
        for passage, ref, _ in read_test_sdp():
            self.convert_and_evaluate(passage, ref)

    def test_split(self):
        """Test that splitting a single-sentence SDP graph converted to UCCA returns the same SDP graph"""
        for passage, ref, _ in read_test_sdp():
            sentences = split2sentences(passage)
            self.assertEqual(len(sentences), 1, "Should be one sentence: %s" % passage)
            sentence = sentences[0]
            self.convert_and_evaluate(sentence, ref)

    def convert_and_evaluate(self, passage, ref):
        converted = to_sdp(passage)
        scores = evaluate(converted, ref)
        self.assertAlmostEqual(scores.average_f1(), 1, msg="\n".join(converted))


class EvaluationTests(unittest.TestCase):
    """Tests the evaluation module functions and classes."""

    def test_evaluate(self):
        """Test that comparing an SDP graph against itself returns perfect F1"""
        for _, ref, sdp_id in read_test_sdp():
            scores = evaluate(ref, ref)
            self.assertAlmostEqual(scores.average_f1(), 1)


def read_test_sdp():
    with open("test_files/20001001.sdp") as f:
        yield from from_sdp(f, "20001001", return_original=True)
