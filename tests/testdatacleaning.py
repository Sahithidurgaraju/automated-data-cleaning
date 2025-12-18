import os
import pandas as pd
import logging
from src.pandasdatacleaning import Datacleaner
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

# @pytest.mark.tc_0001
# def test_data_shape(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df,csvname)
#     rows, cols = cleaner.df.shape
#     logger.info(f"Rows and columns of given data:{cleaner.df.shape}")
#     assert rows>0, "rows are empty"
#     assert cols>0, "columns are empty"

# @pytest.mark.tc_0002
# def test_get_columns(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df)
#     columns = cleaner.get_column_names(csvname,logger)
#     logger.info(f"columns:\n{columns}")
#     #checking the column names are present or not
#     assert isinstance(columns, (list,pd.Index)),"Column names should be a list or Index"

# @pytest.mark.tc_0003
# def test_column_name_issues_present(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df)
#     cleaner_issues = cleaner.check_column_has_issues(csvname,logger)
#     logger.info(f"issues found:\n{cleaner_issues}")
#     #checking is there any issues
#     assert cleaner_issues is True

# @pytest.mark.tc_0004
# def test_column_name_issues(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df)
#     count = 0
#     columns = cleaner.check_column_name_issues()
#     for column, issues in columns.items():
#         if issues:
#             count += 1
#     logger.info(f"Total issues found:{count}")
#     cleaner_issues = cleaner.detect_column_name_issues_save_json(csvname,df,logger)
#     logger.info(f"column name issues:\n{cleaner_issues}")
#     #checking what are the issues present in the column names
#     assert os.path.exists(cleaner_issues), "json file not created"
# @pytest.mark.tc_0005
# def test_column_type_issues(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df)
#     columns = cleaner.detect_column_types_save_json(csvname,df,logger)
#     cleaner_column_types = cleaner.apply_schema_from_json(csvname,df,logger)
#     logger.info(f"detected column types saved to {columns}")
#     # assert os.path.exists(columns), "json file not created"
# @pytest.mark.tc_0006
# def test_print_head_data(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df)
#     head_df = cleaner.df.head(4)
#     logger.info(f"\n{head_df}")
#     assert head_df.notna().sum().sum() > 0, "Head contains only missing values"
# @pytest.mark.tc_0007
# def test_missing_values(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df) 
#     columns = cleaner.get_column_names() 
#     missing_values = cleaner.df.isnull().sum()
#     missing_values = missing_values[missing_values>0]
#     file_path=cleaner.plot_missing_values(csvname,df,logger)
#     logger.info(f"Test case for {csvname}: missing values plot created at {file_path}")   
#     # checking
#     assert not missing_values.empty,"No missing values found in any column"
#     assert file_path is not None, "Function returned None even though missing values exist"
#     assert os.path.exists(file_path), "Plot file not created"
# @pytest.mark.tc_0008
# def test_export_data(messy_data):
#     csvname,df,logger=messy_data
#     cleaner = Datacleaner(df) 
#     cleaned_df = cleaner.standardize_columns(csvname,df,logger)
#     # cleaner.export_to_csv(csvname=csvname, df_to_export=cleaned_df)
#     logger.info(f"log exported{cleaned_df}")
@pytest.mark.tc_0009
def test_pipeline(messy_data):
    csvname,df,logger=messy_data
    cleaner = Datacleaner(df,csvname) 
    load_data = cleaner.load_csv(csvname,logger=logger)
    cleaner.plot_outliers(csvname, logger,when="before", cleanup_old=True)
    cleaner.plot_missing_values(csvname, df, logger, show_plot=False,when="before",cleanup_old=True)
    
    logger.info(f"[{csvname}] BEFORE-cleaning complete\n{'-'*50}")
    # # # # # cleaning   
    
    # # # logger.info("Entering standard_data") 
    cleaner.standard_data(csvname,logger)
    cleaned_df = cleaner.handle_missing(csvname,logger,strategy="auto")
    # logger.info(f"{cleaned_df}")
    cleaner.detect_and_save_schema(cleaned_df, csvname,stage="after", logger=logger)
    cleaner.apply_schema_from_json(csvname,stage="after",logger=logger)
    cleaner.remove_irrelevant_columns(csvname,logger)
    cleaner.plot_outliers(csvname,logger,when="after", cleanup_old=True)
    cleaned_df = cleaner.get_cleaned_data(csvname,logger)
    # # # logger.info(f"{cleaned_df.head(10)}")
    # # # # Plot missing values after cleaning
    cleaner.plot_missing_values(csvname, cleaned_df, logger, when="after",show_plot=False)
    

    # # # # # # # Export cleaned CSV
    export_path = cleaner.export_to_csv(csvname, df_to_export=cleaned_df)

    logger.info(f"[{csvname}] AFTER-cleaning complete. Cleaned CSV saved at: {export_path}")
    logger.info(f"[{csvname}] Cleaned DataFrame preview")
