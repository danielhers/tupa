"""Testing code for the tupa.parser module, unit-testing only."""

import pytest

from tupa.config import Config
from tupa.parse import Parser


@pytest.fixture
def config():
    c = Config("test_files/output.mrp")
    c.update({"conllu": "test_files/udpipe.mrp", "alignment": "test_files/isi.mrp",
              "lstm_layer_dim": 4, "max_edge_labels": 3, "max_lemmas": 3, "max_node_labels": 3,
              "max_node_properties": 3, "max_words": 3, "curriculum": True, "iterations": 1})
    return c


def test_parse(config):
    parser = Parser(config=config)
    list(parser.train(config.args.input))
    list(parser.parse(config.args.input))
