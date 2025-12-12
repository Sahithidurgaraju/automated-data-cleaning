import pandas as pd
import warnings
import numpy as np
import re
import os
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import time
import glob
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Datacleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.logs = []
    
    def __str__(self):
        return str(self.df)
    def reload_file(self,filepath):
        self.df = pd.read_csv(filepath)
        self.logs.append("CSV reloaded from given path")
    #displaying displaying column names
    def get_column_names(self):
        columns = self.df.columns
        self.logs.append(f"Displaying columns:{columns}")
        return columns
    
    #handle missing values in dataset
    def handle_missing(self,strategic="auto",fill_value=None):
        """
        Docstring for handle_missing
        
        :param df: dataframe of data
        :param strategy: 'auto' -> numeric = median, categorical = mode
              'drop' -> drop rows with missing values
              'fill' -> fill with specified value
        """

        if strategic == "drop":
            before_rows = self.df.shape[0]
            self.df.dropna(inplace = True)
            after_rows = self.df.shape[0]
            self.logs.append(f"Dropped {before_rows-after_rows}rows with missing values")      

        elif strategic == "fill":
            if fill_value is not None:
                before_rows = self.df.shape[0]            
                self.df.fillna(fill_value, inplace=True)
                after_rows = self.df.shape[0]
                self.logs.append(f"filled missing values with {fill_value}")
            else:
                self.logs.append(f"fillvalue is not provided")
        
        elif strategic == "auto":
            #if any Nan row present in data
            before_columns = self.df.shape[1]
            self.df.dropna(axis=1, how="all",inplace = True)
            after_columns = self.df.shape[1]
            removed = before_columns-after_columns
            if removed:
                self.logs.append(f"Dropped {removed}columns with entire Null values") 
            else:
                print("No fully empty rows to delete")
            for col in self.df.columns:
                #skip the non missing value 
                if self.df[col].isnull().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = None
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)                    
                        median_val = self.df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    self.df[col].fillna(median_val, inplace=True)
                    self.logs.append(f"Filled missing numeric column '{col}' with median: {median_val}")
                elif pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                    mode_vals = self.df[col].mode()
                    if not mode_vals.empty:
                        mode_val = mode_vals[0]
                    else:
                        mode_val = "Unkown"
                    self.df[col].fillna(mode_val, inplace=True)
                    self.logs.append(f"Filled missing categorical '{col}' with mode: {mode_val}")
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    mode_vals = self.df[col].mode()
                    if not mode_vals.empty:
                        mode_val = mode_vals[0]
                    else:
                        mode_val = pd.Timestamp.min
                    self.df[col].fillna(mode_val, inplace=True)
                    self.logs.append(f"Filled missing datetime column '{col}' with value: {mode_val}")
                else:
                    self.logs.append(f"Skipped column '{col}' (unhandled dtype)")
        else:
            self.logs.append(f"Unknown strategy: {strategic}")

        return self
        
    #remove_duplicates
    def remove_duplicates(self, subsets=None):
        before_rows = self.df.shape[0]
        if subsets is None:
            self.df.drop_duplicates(inplace=True)
        else:
            self.df.drop_duplicates(subset=subsets, inplace=True)
        after_rows = self.df.shape[0]
        removed = before_rows-after_rows
        self.logs.append(f"Removed {removed} duplicate rows")
        return self
    
    #check column names issues to make clean header
    def check_column_name_issues(self):
        issues = {}
        columns = self.get_column_names()
        for col in columns:
            col_issues = []
            if " " in col:
                col_issues.append("contains space")
            if col!= col.strip():
                col_issues.append("leading or trailing whitespace")
            if not re.match(r"^[A-Za-z0-9_]+$", col):
                col_issues.append("special characters found")
            if col.lower() != col:
                col_issues.append("contains upper case letters")
            if col == "":
                col_issues.append("empty column name")
            try:
                col.encode("ascii")
            except UnicodeEncodeError:
                col_issues.append("unicode characters")
            if col_issues:
                issues[col] = col_issues
        return issues
    
    #just checking the column has any issues
    def check_column_has_issues(self):
        return bool(self.check_column_name_issues())
    
    #cleaning the column names 
    def standardize_columns(self):
        new_columns = []
        columns = self.get_column_names()
        for i, col in enumerate(columns):
            original_col = col
            if col.strip() == " ":
                col = f"column_{i+1}"
            col = col.strip()
            col = re.sub(r"[ -]+", "_", col)
            col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
            col = re.sub(r"[^A-Za-z0-9_]", "", col)
            col = col.lower()
            new_columns.append(col)
        self.logs.append(f"Column '{original_col}' standardized to '{col}'")
        self.df.columns = new_columns
        return self
    #handle_outliers
    def handle_outliers(self):
        columns = self.get_column_names()
        for col in columns.select_dtypes(["float", "int"]):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3-Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df[col] = self.df[col].clip(lower, upper)
    def remove_irrevalant_columns(self, threshold = 0.8):
        n_rows = self.df.shape[0]
        for col in self.df.columns:
            if self.df[col].isnull().sum() / n_rows > threshold:
                self.df.drop(columns = [col], inplace=True)
                self.logs.append(f"Removed column '{col}' with >{threshold*100}% missing values")
    def get_cleaned_data(self):
        return self.df.copy()
    
    def export_to_csv(self,file_name=None):
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"cleaned_data_{timestamp}"
        self.df.to_csv(file_name, index=False)
        self.logs.append(f"Data Exported to csv:{file_name}")
        return file_name
    
    def export_to_excel(self,file_name=None):
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"cleaned_data_{timestamp}"
        self.df.to_excel(file_name, index=False)
        self.logs.append(f"Data Exported to excel:{file_name}")
        return file_name
    
    def plot_missing_values(self, outputdir="output", show_plot=False):
        # creating output directory if not exists
        outputdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(outputdir, exist_ok=True)
        #checking missing values
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if missing_values.empty:
            self.logs.append("No missing values found.")
            return None
        #remove old images
        old_files = glob.glob(os.path.join(outputdir, "missing_values_*.png"))
        if old_files:
            for f in old_files:
                os.remove(f)
        #plotting the bar graph of missing values present in the data
        plt.figure(figsize=(8,5))
        ax=sns.barplot(x=missing_values.index, y=missing_values.values, palette='Reds')
        plt.title('Missing Values per Column')
        plt.ylabel('Count of Missing Values')
        plt.xlabel('Columns')
        plt.xticks(rotation=45)
        plt.tight_layout()
        for p in ax.patches:
            value = int(p.get_height())
            ax.text(
        p.get_x() + p.get_width() / 2,   # X position (middle of bar)
        p.get_height(),                  # Y position (top of bar)
        str(value),                      # Text to display
        ha='center',                     # Horizontal align
        va='bottom',                     # Vertical align
        fontsize=10,
        fontweight='bold'
    )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"missing_values_{timestamp}.png"
        save_path = os.path.join(outputdir, filename)
        plt.savefig(save_path)
        plt.close()
        return save_path