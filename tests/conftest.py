from itertools import combinations

import pytest

FORMATS = ("ucca", "amr", "conllu", "sdp")


def pytest_addoption(parser):
    parser.addoption("--write-oracle-actions", action="store_true", help="overwrite oracle action expected results")
    parser.addoption("--multitask", action="store_true", help="test multitask parsing")


@pytest.fixture
def write_oracle_actions(request):
    return request.config.getoption("--write-oracle-actions")


def pytest_generate_tests(metafunc):
    if "formats" in metafunc.fixturenames:
        metafunc.parametrize("formats",
                             [c for n in range(1, (3 if metafunc.config.getoption("--multitask") else 1) + 1)
                              for c in combinations(FORMATS, n)], ids="-".join)
