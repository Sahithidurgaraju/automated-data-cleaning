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

    df = pd.read_csv(
        file_path,
        na_values=["", "NA", "N/A", "None", "null", "-", "?"],
        keep_default_na=True
    )

    return csv_name, df, logger

#here onwards testing of before cleaning of messy data starts:
#testing column name has any issues like white spaces, unicode characters, upper characters

@pytest.mark.tc_0001
def test_load_csv(messy_data):
    csvname,df,logger = messy_data
    cleaner = Datacleaner(df,csvname)
    load_data = cleaner.load_csv(csvname,logger)
    logger.info(f"[{csvname}]-Loaded data")
    text_columns = load_data.select_dtypes(include="object").columns
    for col in text_columns:
        logger.info(f"{col}")
        assert not load_data[col].str.startswith(" ").any()
        assert not load_data[col].str.endswith(" ").any()
    assert load_data is not None
    assert not load_data.empty
    assert load_data.shape[0] > 0

@pytest.mark.tc_0002
def test_shape(messy_data):
    csvname,df,logger = messy_data
    cleaner = Datacleaner(df,csvname)
    shape,rows,columns = cleaner.shape(csvname,logger)
    logger.info(f"shape of the data:{shape}")
    assert isinstance(shape, tuple), "Shape should be a tuple"
    assert len(shape) == 2, "Shape should have (rows, columns)"
    assert rows > 0, "Row count should be > 0"
    assert columns > 0, "Column count should be > 0"

@pytest.mark.tc_0003
def test_handle_missing(messy_data):
    csvname,df,logger = messy_data
    cleaner = Datacleaner(df,csvname)
    missing_data = cleaner.handle_missing(csvname,logger,strategy="auto")
    for col in missing_data.columns:
        if cleaner.is_structural_na_column(col, missing_data[col]):
            continue

        # Only assert for non-structural columns
        null_ratio = missing_data[col].isna().mean()

        # Allow reasonable threshold (configurable)
        assert null_ratio <= 0.3, (
            f"Column '{col}' has high null ratio {null_ratio:.1%}"
        )
@pytest.mark.tc_0004
def test_remove_duplicates(messy_data):
    csvname,df,logger = messy_data
    cleaner = Datacleaner(df,csvname)
    cleaner.remove_duplicates(csvname,logger)
    dedup_keys = cleaner.derive_duplicate_subset(csvname)
    if cleaner.deduplication_status == "executed":
        assert df.duplicated(subset=dedup_keys).sum() == 0
    elif cleaner.deduplication_status == "aborted":
        assert df.shape[0] > 0 

@pytest.mark.tc_0005
def test_pipeline(messy_data):
    csvname,df,logger=messy_data
    cleaner = Datacleaner(df,csvname) 
    df_clean = cleaner.datacleaning_pipeline(csvname=csvname,cleanup_old=True,strategy="auto",show_plot=False,logger=logger) 
    logger.info(f"[{csvname}] AFTER-cleaning complete. Cleaned CSV saved ")
    logger.info(f"[{csvname}] Cleaned DataFrame preview")
    assert not df_clean.empty
    assert len(df_clean) > 0
    
@pytest.mark.tc_0006
def test_validation_report_structure(messy_data):
    csvname, df, logger = messy_data
    cleaner = Datacleaner(df, csvname)
    rows_before = len(df)
    cleaner.datacleaning_pipeline(csvname=csvname,cleanup_old=True,strategy="auto",show_plot=False,logger=logger)

    report = cleaner.validate_cleaned_data(csvname, logger,df,rows_before)
    dedup_check = report["checks"]["deduplication"]
    # basic structure
    assert isinstance(report, dict)
    assert "dataset" in report
    assert "status" in report
    assert "checks" in report
    assert isinstance(report["checks"], dict)
    assert report["dataset"] == csvname
    assert report["status"] in ("PASS", "FAIL")
    #checking report status 
    for check, result in report["checks"].items():
        assert result["status"] in ("PASS", "FAIL")
        assert "message" in result    
    #deduplication status 
    if cleaner.deduplication_status == "aborted":
        assert dedup_check["status"] == "PASS"
        assert "skipped" in dedup_check["message"].lower()
    elif cleaner.deduplication_status == "executed":
        assert dedup_check["status"] == "PASS"



