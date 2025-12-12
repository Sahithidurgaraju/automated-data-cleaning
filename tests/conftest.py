import pytest
from src.logger_config import setup_logging

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Automatically setup logging for all tests."""
    setup_logging()
