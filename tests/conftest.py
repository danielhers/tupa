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
        formats = [[f] for f in FORMATS]
        if metafunc.config.getoption("--multitask"):
            formats += [[FORMATS[0], f] for f in FORMATS[1:]]
        metafunc.parametrize("formats", formats, ids="-".join)
