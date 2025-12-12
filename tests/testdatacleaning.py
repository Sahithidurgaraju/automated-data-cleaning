import os
import pandas as pd
import logging
from src.pandasdatacleaning import Datacleaner
import warnings
import re
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)  
pd.set_option('display.max_rows', None)  # show all rows
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
#data folder and file name should given after the project root
def messy_data():
    #filepath
    file_path = os.path.join(PROJECT_ROOT, "data", "messydata.csv")
    df = pd.read_csv(
    file_path,
    na_values=["", " ", "  ", "   ", "\t", "\n", "NA", "N/A", "n/a", "NONE", "None", "null", "NULL", "-", "?", "__"],
    keep_default_na=True
)
    return df.copy()

#here onwards testing of before cleaning of messy data starts:
#testing column name has any issues like white spaces, unicode characters, upper characters

@pytest.mark.tc_0001
def test_get_columns(messy_data):
    cleaner = Datacleaner(messy_data)
    columns = cleaner.get_column_names()
    logging.info(f"columns:\n{columns}")
    #checking the column names are present or not
    assert isinstance(columns, (list,pd.Index)),"Column names should be a list or Index"

@pytest.mark.tc_0002
def test_column_name_issues_present(messy_data):
    cleaner = Datacleaner(messy_data)
    cleaner_issues = cleaner.check_column_has_issues()
    logging.info(f"issues found:\n{cleaner_issues}")
    #checking is there any issues
    assert cleaner_issues is True

@pytest.mark.tc_0003
def test_column_name_issues(messy_data):
    cleaner = Datacleaner(messy_data)
    count = 0
    cleaner_issues = cleaner.check_column_name_issues()
    columns = cleaner.get_column_names()
    for column_name, issue in cleaner_issues.items():        
        if issue:
            count += 1
            logging.info(f"issues in columns:\n{column_name}:{issue}\n")
    logging.info(f"Total issues found:{count}")
    #checking what are the issues present in the column names
    assert isinstance(cleaner_issues,dict)
    assert cleaner_issues !={}
    assert len(cleaner_issues)>0
    assert any(any(ch.isupper() for ch in col) for col in columns), \
    "Expected messy column names with uppercase, but all columns are lowercase"
    # assert all(" " not in col for col in columns), "some columns contain spaces"
    # assert all(re.match(r"^[a-z0-9_]+$", col) for col in columns), "Some columns have invalid characters"
    for issues_list in cleaner_issues.values():
        assert isinstance(issues_list, list)

@pytest.mark.tc_0004
def test_print_head_data(messy_data):
    cleaner = Datacleaner(messy_data)
    head_df = cleaner.df.head(4)
    logging.info(f"\n{head_df}")
    assert head_df.notna().sum().sum() > 0, "Head contains only missing values"

@pytest.mark.tc_0005
def test_data_shape(messy_data):
    cleaner = Datacleaner(messy_data)
    rows, cols = cleaner.df.shape
    logging.info(cleaner.df.shape)
    assert rows>0, "rows are empty"
    assert cols>0, "columns are empty"

@pytest.mark.tc_0006
def test_missing_values(messy_data,tmp_path):
    cleaner = Datacleaner(messy_data) 
    outputdir = tmp_path / "output"
    columns = cleaner.get_column_names() 
    missing_values = cleaner.df.isnull().sum()
    missing_values = missing_values[missing_values>0]
    file_path=cleaner.plot_missing_values(outputdir=str(outputdir))
    logging.info(file_path)
    logging.info(f"missing values found in these columns:\n{missing_values}")    
    # checking
    assert not missing_values.empty,"No missing values found in any column"
    assert file_path is not None, "Function returned None even though missing values exist"
    assert os.path.exists(file_path), "Plot file not created"



