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
from src.config import DATA_DIR,JSON_DIR
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
DATA_DIR = PROJECT_ROOT / "data"
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
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {DATA_DIR}")
        file_path = csv_files[0]

    csv_name = file_path.stem
    logger = get_logger(csv_name)
    logger.info(f"Processing: {file_path}")
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:

            df = pd.read_csv(
        file_path,
        na_values=["", "NA", "N/A", "None", "null", "-", "?","Nan", "Inf", "Not Applicable"],
        sep=None,          
    engine="python", encoding=enc,           
    keep_default_na=True,
    skip_blank_lines=True
)
            if logger:
                logger.info(f"Loaded CSV using encoding: {enc}")
            break
        except UnicodeDecodeError:
                continue

    return csv_name, df, logger
#here onwards testing of before cleaning of messy data starts:
#testing column name has any issues like white spaces, unicode characters, upper characters

@pytest.mark.tc_0004
def test_pipeline(messy_data):
    csvname,df,logger=messy_data
    cleaner = Datacleaner(df,csvname) 
    df_clean = cleaner.datacleaning_pipeline(csvname=csvname,cleanup_old=True,strategy="auto",show_plot=False,logger=logger) 
    logger.info(f"[{csvname}] AFTER-cleaning complete. Cleaned CSV saved ")
    logger.info(f"[{csvname}] Cleaned DataFrame preview")
    assert not df_clean.empty
    assert len(df_clean) > 0