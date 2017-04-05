"""Testing code for the amr package, unit-testing only."""

import unittest

from contrib import amrutil, convert


class ConversionTests(unittest.TestCase):
    """Tests convert module correctness and API."""

    def test_convert(self):
        """Test that converting an AMR to UCCA and back retains perfect Smatch F1"""
        for passage, ref, amr_id in read_test_amr():
            converted = convert.to_amr(passage)[0]
            scores = amrutil.evaluate(converted, ref, amr_id)
            self.assertAlmostEqual(scores.f1, 1, msg=converted)


class UtilTests(unittest.TestCase):
    """Tests the amrutil module functions and classes."""

    def test_evaluate(self):
        """Test that comparing an AMR against itself returns perfect Smatch F1"""
        for _, ref, amr_id in read_test_amr():
            scores = amrutil.evaluate(ref, ref, amr_id)
            self.assertAlmostEqual(scores.f1, 1)


def read_test_amr():
    with open("test_files/LDC2014T12.txt") as f:
        return list(convert.from_amr(f, return_amr=True))
