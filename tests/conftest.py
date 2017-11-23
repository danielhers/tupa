import pytest

def pytest_addoption(parser):
    parser.addoption("--write-oracle-actions", action="store_true", help="overwrite oracle action expected results")


@pytest.fixture
def write_oracle_actions(request):
    return request.config.getoption("--write-oracle-actions")