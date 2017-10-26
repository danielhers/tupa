"""Testing code for the conllu format, unit-testing only."""

import unittest

from ucca.convert import split2sentences

from scheme.conversion.conllu import from_conllu, to_conllu
from scheme.evaluation.conllu import evaluate


class ConversionTests(unittest.TestCase):
    """Tests conversion module correctness and API."""

    def test_convert(self):
        """Test that converting an Universal Dependencies tree to UCCA and back retains perfect LAS F1"""
        for passage, ref, _ in read_test_conllu():
            self.convert_and_evaluate(passage, ref)

    def test_split(self):
        """Test that splitting a single-sentence Universal Dependencies tree converted to UCCA returns the same tree"""
        for passage, ref, _ in read_test_conllu():
            sentences = split2sentences(passage)
            self.assertEqual(len(sentences), 1, "Should be one sentence: %s" % passage)
            sentence = sentences[0]
            self.convert_and_evaluate(sentence, ref)

    def convert_and_evaluate(self, passage, ref):
        converted = to_conllu(passage)
        scores = evaluate(converted, ref)
        self.assertAlmostEqual(scores.average_f1(), 1, msg="\n".join(converted))


class EvaluationTests(unittest.TestCase):
    """Tests the evaluation module functions and classes."""

    def test_evaluate(self):
        """Test that comparing an Universal Dependencies graph against itself returns perfect LAS F1"""
        for _, ref, conllu_id in read_test_conllu():
            scores = evaluate(ref, ref)
            self.assertAlmostEqual(scores.average_f1(), 1)


def read_test_conllu():
    with open("test_files/UD_English.conllu") as f:
        yield from from_conllu(f, "weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000",
                               return_original=True)
    with open("test_files/UD_German.conllu") as f:
        yield from from_conllu(f, "train-s10000", return_original=True)
