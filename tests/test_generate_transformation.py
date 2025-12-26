import os
import pandas as pd
import logging
from src.pandasdatacleaning import Datacleaner
from src.datatransformation import Datatranformer
import warnings
import re
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from src.logger_config import get_logger 
matplotlib.use("Agg")
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=1, progress_bar=False)
from src.config import DATA_DIR,JSON_DIR,CLEANED_DATA_DIR
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  
pd.set_option('display.max_rows', None)  # show all rows
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
from pathlib import Path
import pandas as pd
import pytest

# Assuming get_logger is defined somewhere
# from your_logging_module import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data_output"
LOG_DIR = PROJECT_ROOT/"logs"

@pytest.fixture
def messy_data():
    """
    Fixture to load a CSV for testing:
    - Uses CSV from PYTEST_CURRENT_CSV if set
    - Otherwise, falls back to the first CSV in the data folder
    Returns:
        csv_name: str
        df: pd.DataFrame
        logger: logger instance
    """
    current_csv = os.environ.get("PYTEST_CURRENT_CSV")
    
    if current_csv:
        file_path = Path(current_csv)
    else:
        # Fallback to first CSV in data folder
        csv_files = list(CLEANED_DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {CLEANED_DATA_DIR}")
        file_path = csv_files[0]

    csv_name = file_path.stem
    logger = get_logger(csv_name)
    logger.info(f"Processing: {file_path}")

    df = pd.read_csv(file_path)

    return csv_name, df, logger

#here onwards testing of before cleaning of messy data starts:
#testing column name has any issues like white spaces, unicode characters, upper characters

@pytest.mark.tc_0001
def test_save_transformation(messy_data):
    csvname,df,logger=messy_data
    cleaner = Datacleaner(df,csvname) 
    transformer = Datatranformer(df,csvname)
    json_path = transformer.save_config_file(df,csvname,logger)
    assert json_path is not None, "JSON path returned None!"
    assert os.path.exists(json_path), f"JSON config file was NOT created: {json_path}"
    assert Path(json_path).is_file(), f"Expected JSON file but got directory: {json_path}"
    assert os.path.getsize(json_path) > 0, f"JSON file is empty: {json_path}"