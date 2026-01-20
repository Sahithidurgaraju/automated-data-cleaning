import os
import pandas as pd
import logging
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
from src.config import DATA_DIR,JSON_DIR,CLEANED_DATA_DIR,DASHBOARD_DIR
from src.pandasdatacleaning import Datacleaner
from src.etldashboard import Datadashboard
from src.datatransformation import Datatranformer
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  
pd.set_option('display.max_rows', None)  # show all rows
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Assuming get_logger is defined somewhere
# from your_logging_module import get_logger
LOG_DIR = PROJECT_ROOT/"logs"
DATA_DIR = PROJECT_ROOT/"data"
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

    clean_name = Path(file_path).stem.lower().replace(" ", "_") + "_cleaned"
    csv_name = clean_name
    plot_csv = Path(file_path).name
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

    return plot_csv, csv_name, df, logger
#here onwards testing of before cleaning of messy data starts:
#testing column name has any issues like white spaces, unicode characters, upper characters

@pytest.mark.tc_0001
def test_save_pdf_metrics(messy_data):
    plot_csv, csvname,df,logger=messy_data
    cleaner = Datacleaner(df,csvname) 
    dashboard = Datadashboard()
    tables = dashboard.get_tables_from_database("test")
    dbname = "test"
    base = "transformed_" + csvname.lower()
    matched = [t for t in tables if base in t.lower()]
    if not matched:
        logger.error(f"No MySQL table contains `{csvname}` in its name")
        return
    table_name = base
    df_after= dashboard.load_table(dbname, table_name)
    logger.info(f"{df.head()}")
    df_before = pd.read_csv(
    DATA_DIR / plot_csv,
    sep=None,                    # auto-detect ; , \t
    engine="python",             # REQUIRED for messy CSVs
    encoding="latin1",           # handles � Ê ë etc.
    keep_default_na=True,
    skip_blank_lines=True
)
    logger.info(f"{plot_csv}")
    data, dedupe_status = cleaner.remove_duplicates(plot_csv,logger)
    logger.info(f"{dedupe_status}")
    summary_df,quality_score = dashboard.cleaning_score(df_before,df_after,dedupe_status,plot_csv,dbname,table_name)
    summary_df.loc[
        summary_df["Metric"] == "Overall Cleaning improvement Score",
            "Improved (%)"
        ].values[0]
    summary_df.loc[
        summary_df["Metric"] == "Final Data Quality",
            "Improved (%)"
        ].values[0]
    summary_df.loc[
        summary_df["Metric"] == "Rows",
            "Improved (%)"
        ].values[0]
    summary_df.loc[
        summary_df["Metric"] == "Columns",
            "Improved (%)"
        ].values[0]
    insights = [
        f"Final Data Quality Score: {quality_score}",
        f"Rows Before: {len(df_before)}",
        f"Rows After: {len(df_after)}",
        "Vectorized transformations applied",
        "Validation rules enforced"
    ]

    pdf_path = dashboard.generate_pdf(
        csvname,
        summary_df,
        quality_score,
        insights,
        output_dir=DASHBOARD_DIR
    )

    logger.info(f"PDF generated at {pdf_path}")
    assert os.path.exists(pdf_path), "PDF file not created"
