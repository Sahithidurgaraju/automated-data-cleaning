# tests/conftest.py
import pytest
from src.logger_config import get_logger  # import your logger setup
from src.config import DATA_DIR

@pytest.fixture
def csv_logger(request):
    """
    Fixture to provide a CSV-specific logger to each test case.
    You can optionally parametrize the CSV name.
    """
    # Get CSV name from param if provided
    csv_name = getattr(request, 'param', None)

    if csv_name is None:
        # Default to first CSV in DATA_DIR
        csv_file = list(DATA_DIR.glob("*.csv"))[0]
        csv_name = csv_file.stem

    # Return the logger for this CSV
    return get_logger(csv_name)

# tests/conftest.py
def pytest_configure(config):
    # Enable logger output in HTML
    config.option.log_cli = True
    config.option.log_cli_level = "INFO"

    # Add custom metadata
    if hasattr(config, "_metadata"):
        # Optional: clear default environment
        config._metadata.clear()
        # config._metadata["Author"] = "Your Name"
        config._metadata["Project"] = "Data Cleaning"
        config._metadata["Version"] = "1.0"
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_html_results_summary(prefix, summary, postfix):
    prefix.extend([f"<p style='font-size:smaller;'>programmer: DolphinsCoderz</p>"])

