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
from sqlalchemy import create_engine


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
def test_transformation(messy_data):
    csvname,df,logger=messy_data
    transformer = Datatranformer(df,csvname)
    df_c,report= transformer.apply_transformations(df,csvname,logger)
    assert df_c is not None, "Transformation returned None!"
    assert not df_c.empty, "Transformed DataFrame is empty!"
    assert isinstance(df_c, pd.DataFrame), "Output is not a DataFrame"
    logger.info(f"{df_c.head(10)}")
    if not df_c.filter(like="_bin").empty:
        file = transformer.plot_all_bins(df_c,csvname,logger)
        assert file.is_file(), "Bin plot was not saved as file!"
        logger.info("Bin plotting successful")
    else:
        logger.info("No bins to plot — skipping (acceptable)")
    
@pytest.mark.tc_0002
def test_push_to_sql(messy_data):
    csvname,df,logger=messy_data
    transformer = Datatranformer(df,csvname)
    len_df = len(df)
    path = transformer.get_database_credentials(logger)
    dbname,table,db_engine = transformer.push_to_sql(df,logger,path,csvname)
    null_count, row_count = transformer.test_sql_data(dbname, table, db_engine, logger)
    logger.info("successfully pushed data to sql")
    assert isinstance(row_count, int), "Row count is not integer!"
    assert isinstance(null_count, int), "Null count is not integer!"
    assert row_count > 0, "No rows inserted into SQL!"
    assert row_count == len_df, "Row count mismatch after push!"

