import pytest

# Mark all tests in this directory to run forked
def pytest_collection_modifyitems(items):
    for item in items:
        # Check if test is in this directory
        if "z3" in str(item.fspath):
            item.add_marker(pytest.mark.forked)

# Configure logging for forked tests
def pytest_configure(config):
    config.option.log_cli = True
    config.option.log_cli_level = "INFO"
    config.option.log_cli_format = "%(message)s"
