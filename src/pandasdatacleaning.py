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
from pathlib import Path
import json


TEXT_DIRTY_PATTERN = r"\[[^\]]*\]|[†‡*]|[\x00-\x1F\x7F]"
NUMERIC_PATTERN = r"[\$\€\£\₹(),]|\d+[KMBkmb]$"
YEAR_PATTERN = r"(19\d{2}|20\d{2})"
YEAR_RANGE_PATTERN = r"(19\d{2}|20\d{2}).*(19\d{2}|20\d{2})"


TEXT_THRESHOLD = 0.6
YEAR_THRESHOLD = 0.6
NUMERIC_THRESHOLD = 0.1

    
ID_NAME_KEYWORDS = (
    "id", "uuid", "guid", "hash", "code", "ref", "reference"
)

from src.config import DATA_DIR, JSON_DIR,CLEANED_DATA_DIR,PLOTS_DIR
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Datacleaner:
    def __init__(self, df,csvname):
        self.df = df.copy()
        self.logs = []
        self.column_mapping = {}  
        self.csvname = csvname


    # ----------------------------------
    # 1️⃣ FAST COLUMN NAME CLEANER
    # ----------------------------------
    def clean_colnames(self):
        """
        Clean column names:
        - lowercase
        - strip spaces
        - replace non-alphanumeric with _
        - collapse multiple _
        """
        self.df.columns = (
            self.df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )

    # ----------------------------------
    # 2️⃣ LOAD CSV + INITIAL CLEANING
    # ----------------------------------
    def load_csv(self, csv_path, logger=None, encoding="utf-8",errors = "replace"):
        """
        Load CSV safely, clean column names, clean text columns efficiently.
        """

        # -----------------------------
        # Normalize path
        # -----------------------------
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)

        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")

        if not csv_path.is_absolute():
            csv_path = DATA_DIR / csv_path

        if not csv_path.exists():
            msg = f"CSV file not found: {csv_path}"
            if logger:
                logger.error(msg)
            raise FileNotFoundError(msg)

        # -----------------------------
        # Read CSV (encoding fallback)
        # -----------------------------
        try:
            self.df = pd.read_csv(csv_path, na_values=[""],
    keep_default_na=True,encoding=encoding)
            if logger:
                logger.info(f"Loaded CSV: {csv_path} ({encoding})")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="latin1")
            if logger:
                logger.warning(f"Encoding fallback to latin1: {csv_path}")

        # -----------------------------
        # Clean column names
        # -----------------------------
        self.clean_colnames()

        if logger:
            logger.info(f"Cleaned column names: {list(self.df.columns)}")

        # -----------------------------
        # Clean TEXT columns (FAST)
        # -----------------------------
        text_cols = self.df.select_dtypes(include=["object", "string"]).columns

        for col in text_cols:
            s = self.df[col]

    # Fast unicode fixes (vectorized)
            s = (
            s.astype("string")
         .str.replace("â€ ", "", regex=False)
         .str.replace("â€¡", "", regex=False)
         .str.replace("Ã©", "é", regex=False)
    )

    # APPLY ONCE — NOT apply()
            self.df[col] = self.clean_text(s)

            if logger:
                logger.info(f"Cleaned text column: {col}")

        if logger:
            logger.info(f"CSV loading and cleaning completed: {csv_path}")

        return self.df


 

    
    #displaying displaying column names
    def get_column_names(self,csvname,logger):
        csvname=Path(csvname)
        self.df = self.load_file(csvname,logger)
        columns = self.df.columns
        self.logs.append(f"Displaying columns:{columns}")
        return columns
    
    def normalize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
    Convert all missing-like values (strings, blanks) into real NaNs.
    """

        missing_like = [
        "", " ", "  ", "\t", "\n",
        "nan", "NaN", "NAN",
        "null", "NULL",
        "none", "None", "NONE",
        "-", "--", "—","<NA>","   ",'   ','',' '

    ]
        df.replace("", np.nan, inplace=True)
    # Replace exact string matches
        df = df.replace(missing_like, pd.NA)

        return df



    def handle_missing(self, csvname, logger, strategy="auto", fill_value=None):

        if self.df is None:
            logger.warning(f"[{csvname}] No dataframe loaded")
            return None
        df = self.df
        # df = self.df.replace(pd.NA, np.nan)
        df = self.normalize_missing(self.df.copy())
        
    # ===============================
    # DROP STRATEGY
    # ===============================
        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(f"[{csvname}] Dropped {before - len(df)} rows with missing values")
            self.df = df
            return df

    # ===============================
    # FILL STRATEGY
    # ===============================
        if strategy == "fill":
            if fill_value is None:
                logger.warning(f"[{csvname}] fill_value required for fill strategy")
                return df

            self.df = df.fillna(fill_value)
            logger.info(f"[{csvname}] Filled all missing values with {fill_value}")
            return self.df

    # ===============================
    # AUTO STRATEGY (OPTIMIZED)
    # ===============================
        if strategy == "auto":

        # 1️⃣ Drop fully empty columns
            before_cols = df.shape[1]
            df = df.dropna(axis=1, how="all")
            dropped_cols = before_cols - df.shape[1]

            if dropped_cols > 0:
                logger.info(f"[{csvname}] Dropped {dropped_cols} fully empty columns")

        # 2️⃣ Split columns by dtype ONCE
            num_cols = df.select_dtypes(include=["number"]).columns
            dt_cols = df.select_dtypes(include=["datetime"]).columns
            obj_cols = df.select_dtypes(include=["object", "string"]).columns
            cat_cols = df.select_dtypes(include=["category"]).columns
        # 3️⃣ Numeric → median (vectorized)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                medians = df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(medians)
            # Restore integer columns safely
            for col in df.columns:
                    if df[col].isna().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna("Unknown")
            
        # 4️⃣ Datetime → mode or min
            for col in dt_cols:
                if df[col].isna().any():
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else pd.Timestamp.min
                    df[col] = df[col].fillna(fill_val)

        # 5️⃣ Text → mode (fallback to 'Unknown')
            for col in cat_cols:
                df[col] = df[col].astype("object")

            for col in list(obj_cols) + list(cat_cols):
                if df[col].isna().any():
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    df[col] = df[col].fillna(fill_val)
            for col in df.select_dtypes(include=["category"]).columns:
                    df[col] = df[col].astype("object")

                    df.replace("", np.nan, inplace=True)

        # Decide fill value
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"

        # ADD value to categories BEFORE fill
                    if fill_val not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories([fill_val])

        # Now fill works
                    df[col] = df[col].fillna(fill_val)
            logger.info(
                f"[{csvname}] Missing values handled | "
                f"numeric={len(num_cols)}, "
                f"text={len(obj_cols) + len(cat_cols)}, "
                f"datetime={len(dt_cols)}"
        )

            self.df = df
            return df

        logger.warning(f"[{csvname}] Unknown missing value strategy: {strategy}")
        return self.df

        #check column names issues to make clean header
    def check_column_name_issues(self,csvname,logger):
        issues = {}
        columns = self.get_column_names(csvname,logger)
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
                msg = f"[{csvname}] column '{col} issues:{col_issues}"
                logger.warning(msg)
                self.logs.append(msg)
            if not issues:
                msg = f"[{csvname}] No column name issues found"
                logger.info(msg)
                self.logs.append(msg)
        return issues
    
    def detect_column_name_issues_save_json(self,csvname,logger):
        JSON_DIR.mkdir(exist_ok=True)
        csvname=Path(csvname)
        file_output_dir = JSON_DIR / f"{csvname}_output"
        file_output_dir.mkdir(exist_ok=True)
        print(f"file output folder:{file_output_dir}")
            
        for old_file in file_output_dir.glob(f"{csvname}_detect_column_name_issues*.json"):
            try:
                print("Deleting:", old_file.name)
                old_file.unlink()
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")
        issues = self.check_column_name_issues(csvname,logger)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f"{csvname}_detect_column_name_issues{timestamp}.json"
        save_path = file_output_dir / json_filename

        with open(save_path,"w",encoding="utf-8") as f:
            json.dump(issues,f,indent=4)        
        self.logs.append(f"Detected column name issues saved to {save_path}")
        return self.df
    #just checking the column has any issues
    def check_column_has_issues(self,csvname,logger):
        return bool(self.check_column_name_issues(csvname,logger))



    def load_schema(self,csvname):
        JSON_DIR.mkdir(exist_ok=True)
        base = os.path.splitext(os.path.basename(csvname))[0]
        return json.load(open(JSON_DIR / f"{base}_schema_after.json"))


    def get_outlier_columns(self, df, schema):
        outlier_cols = []

        for col, dtype in schema.items():
            col_lower = col.lower()

        # Exclude year-like columns
            if any(k in col_lower for k in ["year", "yr"]):
                continue

            if dtype == "float":
                outlier_cols.append(col)

            elif dtype == "integer":
            # Allow counts, exclude ranks / ids implicitly by schema
                outlier_cols.append(col)

    # Keep only existing numeric columns
        return [c for c in outlier_cols if c in df.columns]


    def plot_outliers(self,csvname,logger,when="before",show_plot=False,cleanup_old=True):
        """
    Plots outliers BEFORE or AFTER cleaning.
    Does NOT mutate self.df.
    """

        if when == "before":
        # Detect numeric-like columns dynamically
            outlier_cols = []
            for col in self.df.columns:
                s = self.df[col]
                if any(k in col.lower() for k in ["year", "yr"]):
                    continue

            # Case 1: already numeric
                if pd.api.types.is_numeric_dtype(s):
                    if pd.api.types.is_bool_dtype(s):
                        continue
                    s_nonnull = s.dropna()
                    if s_nonnull.empty or s_nonnull.nunique() <= 1:
                        continue
                    outlier_cols.append(col)
                    continue
                

            # Case 2: numeric-like strings
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.notna().mean() >= 0.6:
                    outlier_cols.append(col)


            if not outlier_cols:
                logger.info(f"[{csvname}] No numeric-like columns for BEFORE outliers")
                return None

        # Convert to numeric for plotting
            plot_df = self.df[outlier_cols].apply(pd.to_numeric, errors="coerce")

        else:
        # AFTER cleaning → use schema
            schema = self.load_schema(csvname)
            outlier_cols = self.get_outlier_columns(self.df, schema)

            if not outlier_cols:
                logger.info(f"[{csvname}] No columns eligible for AFTER outliers")
                return None

            plot_df = self.df[outlier_cols].astype("float64").copy()


    # ---------- Outlier clipping for visualization ----------
        for col in outlier_cols:
            Q1 = plot_df[col].quantile(0.25)
            Q3 = plot_df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0 or pd.isna(IQR):
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            plot_df[col] = plot_df[col].clip(lower, upper)

    # ---------- Output directory ----------
        PLOTS_DIR.mkdir(exist_ok=True)
        output_dir = PLOTS_DIR / f"{csvname}_outliers"
        output_dir.mkdir(exist_ok=True)

    # ---------- Cleanup ----------
        if cleanup_old:
            for old in output_dir.glob(f"{csvname}_outliers_{when}_*.png"):
                try:
                    old.unlink()
                except Exception:
                    pass

    # ---------- Plot ----------
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df)
        plt.title(f"Outliers ({when}) - {csvname}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        timestamp = time.strftime("%Y%m%d_%H%M")
        plot_file = output_dir / f"{csvname}_outliers_{when}_{timestamp}.png"
        plt.savefig(plot_file)

        if show_plot:
            plt.show()

        plt.close()

        logger.info(f"[{csvname}] Outliers plot ({when}) saved → {plot_file}")
        return self.df, plot_file

    
    def remove_irrelevant_columns(self, csvname, logger, threshold=0.8):
        n_rows = self.df.shape[0]
        for col in self.df.columns:
            missing_ratio = self.df[col].isnull().sum() / n_rows
            if missing_ratio > threshold:
                self.df.drop(columns=[col], inplace=True)
                msg = f"[{csvname}] Removed column '{col}' with {missing_ratio:.2%} missing values"
                logger.info(msg)
                self.logs.append(msg)
        return self.df
    
    def get_cleaned_data(self,csvname,logger):
        if self.df is None:
            msg = f"[{csvname}] No DataFrame available to return."
            logger.warning(msg)
            self.logs.append(msg)
            return None

        msg = f"[{csvname}] Returning cleaned DataFrame with shape {self.df.shape}"
        logger.info(msg)
        self.logs.append(msg)
    
        return self.df.copy()
    
    def export_to_csv(self,csvname=None,df_to_export=None):
        CLEANED_DATA_DIR.mkdir(exist_ok=True)
        df_to_export = df_to_export if df_to_export is not None else self.df
        csvname = Path(csvname).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if csvname:           
            filename = f"{csvname}_cleaned_{timestamp}.csv"
        else:
            filename = f"cleaned_{timestamp}.csv"

        file_output_dir = CLEANED_DATA_DIR / filename
        for old_file in CLEANED_DATA_DIR.glob(f"{csvname}_*.csv"):
            try:
                print("Deleting:", old_file.name)
                old_file.unlink()
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")
        df_to_export.to_csv(file_output_dir, index=False)
        self.logs.append(f"Data Exported to csv:{file_output_dir}")
        return file_output_dir
    
    def export_to_excel(self,file_name=None):
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"cleaned_data_{timestamp}"
        self.df.to_excel(file_name, index=False)
        self.logs.append(f"Data Exported to excel:{file_name}")
        return file_name
    
    def plot_missing_values(self,csvname, df, logger, show_plot=False,when="before",cleanup_old=False):
        PLOTS_DIR.mkdir(exist_ok=True)
        csvname = Path(csvname).stem
        print(self.logs.append(csvname))
        #checking missing values
        # self.df = self.normalize_missing(self.df.copy())
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        logger.info(f"missing values found:{missing_values}")
        self.logs.append(f"Missing values found: {missing_values}")        
        if missing_values.empty:
            logger.info("No missing values found")
            self.logs.append(f"No missing values found in {csvname}.")
            missing_values = pd.Series([0]*len(self.df.columns), index=self.df.columns)
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

        
        file_output_dir = PLOTS_DIR/ f"{csvname}_missing_values_output"
        file_output_dir.mkdir(exist_ok=True)
        print(f"file output folder:{file_output_dir}")
        # Use timestamp for current run
        timestamp = time.strftime("%Y%m%d_%H%M")  # current run timestamp

    # Cleanup old plots from previous runs
        if cleanup_old:
            for old_file in file_output_dir.glob(f"{csvname}_missing_values_*.png"):
                if timestamp not in old_file.name:  # keep current run files
                    try:
                        old_file.unlink()
                        logger.info(f"Deleted old missing value plot: {old_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_file}: {e}")
        png_filename = f"{csvname}_missing_values_{when}_{timestamp}.png"
        save_path = file_output_dir / png_filename
        plt.savefig(save_path)
        if show_plot:
            plt.show()
            plt.close()
        return self.df

    # =====================================================
    # -------- DYNAMIC ID DETECTION (SAFE) ----------------
    # =====================================================
    def _name_looks_like_id(self, col: str) -> bool:
        col = col.lower()
        return (
        col == "id"
        or col.endswith("_id")
        or col in {"uuid", "guid"}
    )


    def _value_looks_like_id(self, s: pd.Series) -> bool:
        s = s.dropna().astype(str)
        if s.empty:
            return False

    # Must contain BOTH letters and digits
        pattern = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9\-_]{8,}$"
        return s.str.match(pattern).mean() >= 0.9


    def _high_uniqueness(self, s: pd.Series) -> bool:
        s = s.dropna()
        if s.empty:
            return False
        return (s.nunique() / len(s)) >= 0.95


    def is_id_column(self, col: str, s: pd.Series) -> bool:
        name_signal = self._name_looks_like_id(col)
        value_signal = self._value_looks_like_id(s)
        uniq_signal = self._high_uniqueness(s)

    # Require at least TWO signals
        return sum([name_signal, value_signal, uniq_signal]) >= 2

    # ===============================
    # FULL DIRTY RATIO (NO SAMPLING)
    # ===============================
    def dirty_ratio(self, s: pd.Series, pattern: str) -> float:
        s = s.dropna().astype(str)
        if s.empty:
            return 0.0
        return s.str.contains(pattern, regex=True).mean()

    # ===============================
    # DATE DETECTION (FULL SCAN)
    # ===============================
    def is_date_column(self, s: pd.Series) -> bool:
        s = s.dropna().astype(str)
        if s.empty:
            return False
        return pd.to_datetime(s, errors="coerce").notna().mean() >= 0.6

    # ===============================
    # SAFE TEXT CLEANER
    # ===============================
    def clean_text(self, s: pd.Series) -> pd.Series:
        s = s.astype("string")

        # Unicode normalization (SAFE)
        s = s.apply(
            lambda x: unicodedata.normalize("NFKC", x)
            if isinstance(x, str) else x
        )

        s = (
            s.str.replace(r"\[[^\]]*\]", "", regex=True)
             .str.replace(r"[†‡*]", "", regex=True)
             .str.replace(r"[\x00-\x1F\x7F]", "", regex=True)
             .str.replace("“", '"', regex=False)
             .str.replace("”", '"', regex=False)
             .str.replace("‘", "'", regex=False)
             .str.replace("’", "'", regex=False)
             .str.replace("–", "-", regex=False)
             .str.replace("—", "-", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
        )

        return s

    # ===============================
    # SAFE NUMERIC CLEANER
    # ===============================
    def clean_numeric(self, s: pd.Series) -> pd.Series:
        raw = s.astype(str)

        s = raw.str.strip()
        s = s.str.replace(r"\[[^\]]*\]", "", regex=True)
        s = s.str.replace(r"^[\$\€\£\₹]", "", regex=True)
        s = s.str.replace(",", "", regex=False)

        multiplier = (
            s.str.extract(r"([KMBkmb])$", expand=False)
             .map({"K":1e3,"k":1e3,"M":1e6,"m":1e6,"B":1e9,"b":1e9})
             .fillna(1)
        )

        s = s.str.replace(r"[KMBkmb]$", "", regex=True)
        s = s.str.replace(r"[^\d\.\-]", "", regex=True)

        num = pd.to_numeric(s, errors="coerce") * multiplier

        # Preserve original if conversion fails
        return num.where(num.notna(), raw)

    # ===============================
    # YEAR EXTRACTION (NON-DESTRUCTIVE)
    # ===============================
    def extract_year(self, s: pd.Series):
        s = s.astype(str)
        s = s.str.replace(r"[–—−]", "-", regex=True)
        s = s.str.replace(r"(\d{4})(?=\d{4})", r"\1-", regex=True)

        years = s.str.findall(YEAR_PATTERN)
        start = years.str[0].astype("Int64")
        end = years.str[-1].where(years.str.len() > 1).astype("Int64")

        return start, end

    # ===============================
    # MAIN FULL CLEAN PIPELINE
    # ===============================
    def standard_data(self, csvname=None, logger=None) -> pd.DataFrame:
        cleaned_cols, skipped_cols = [], []

        for col in self.df.columns:
            s = self.df[col]

            # ---- ID PROTECTION ----
            if self.is_id_column(col,s):
                self.df[col] = s.astype("string").str.strip()
                skipped_cols.append(col)
                continue

            # ---- DIRTY RATIOS (FULL DATA) ----
            text_ratio = self.dirty_ratio(s, TEXT_DIRTY_PATTERN)
            numeric_ratio = self.dirty_ratio(s, NUMERIC_PATTERN)
            year_range_ratio = self.dirty_ratio(s, YEAR_RANGE_PATTERN)
            year_presence_ratio = self.dirty_ratio(s, YEAR_PATTERN)

            # ---- DATE ----
            if self.is_date_column(s) and text_ratio >= TEXT_THRESHOLD:
                self.df[col] = pd.to_datetime(self.clean_text(s), errors="coerce")
                cleaned_cols.append(col)
                continue

            # ---- YEAR ----
            if year_presence_ratio >= YEAR_THRESHOLD and year_range_ratio >= YEAR_THRESHOLD:
                # self.df[f"{col}_raw"] = s
                y1, y2 = self.extract_year(s)
                self.df[col] = y1
                if y2.notna().any():
                    self.df[f"{col}_end"] = y2
                cleaned_cols.append(col)
                continue

            # ---- NUMERIC ----
            if numeric_ratio >= NUMERIC_THRESHOLD:
                self.df[col] = self.clean_numeric(s)
                cleaned_cols.append(col)
                continue

            # ---- TEXT ----
            if self.looks_like_text(s):
                text_dirty_ratio = self.dirty_ratio(s, TEXT_DIRTY_PATTERN)

                if text_dirty_ratio >= TEXT_THRESHOLD:
                    self.df[col] = self.clean_text(s)
                    cleaned_cols.append(col)
                else:
                    self.df[col] = s.astype("string").str.strip()
                    skipped_cols.append(col)

                    continue
        if logger:
            logger.info(f"[{csvname}] Cleaned columns: {cleaned_cols}")
            logger.info(f"[{csvname}] Skipped columns: {skipped_cols}")

        return self.df
    
    def looks_like_text(self, s: pd.Series) -> bool:
        s = s.dropna().astype(str)
        if s.empty:
            return False

    # contains letters
        has_letters = s.str.contains(r"[A-Za-z]", regex=True).mean()

    # purely numeric (bad for text)
        pure_numeric = s.str.match(r"^\d+(\.\d+)?$", na=False).mean()

        return has_letters >= 0.6 and pure_numeric < 0.2
    


    # ==================================================
    # INTERNAL: DYNAMIC COLUMN TYPE DETECTOR
    # ==================================================
    def _detect_column_type(self, s: pd.Series) -> str:
        s_non_null = s.dropna()

        if s_non_null.empty:
            return "text"

        # 1️⃣ BOOLEAN
        lowered = s_non_null.astype(str).str.lower().unique()
        if set(lowered).issubset({"0", "1", "true", "false", "yes", "no"}):
            return "boolean"

        # 2️⃣ NUMERIC (CRITICAL: FIRST)
        num = pd.to_numeric(s_non_null, errors="coerce")
        numeric_ratio = num.notna().mean()

        if numeric_ratio >= 0.95:
            values = num.dropna().to_numpy(dtype="float64", copy=False)
            integer_ratio = (values == np.floor(values)).mean()

            if integer_ratio >= 0.99:
                return "integer"
            return "float"

        # 3️⃣ DATETIME (ONLY IF NOT NUMERIC)
        dt = pd.to_datetime(s_non_null, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.95:
            return "datetime"

        # 4️⃣ CATEGORY (LOW CARDINALITY)
        cardinality_ratio = s_non_null.astype(str).nunique() / len(s_non_null)
        if cardinality_ratio <= 0.1:
            return "category"

        # 5️⃣ TEXT
        return "text"

    # ==================================================
    # DETECT + SAVE SCHEMA (BEFORE / AFTER)
    # ==================================================
    def detect_and_save_schema(self,df: pd.DataFrame,csvname: str,stage: str,logger=None) -> dict:
        """
    Detect schema for a given dataframe and save it to JSON.

    stage: 'before' or 'after'
    """

        JSON_DIR.mkdir(exist_ok=True)
        base = Path(csvname).stem
        schema_path = JSON_DIR / f"{base}_schema_{stage}.json"

    # Delete previous schema for the same stage
        if schema_path.exists():
            schema_path.unlink()

    # Detect schema (NO mutation)
        schema = {
        col: self._detect_column_type(df[col])
        for col in df.columns
    }

    # Save schema
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=4)

        if logger:
            logger.info(f"[{csvname}] {stage.upper()} schema saved → {schema_path}")

        return schema

    # ==================================================
    # APPLY SCHEMA (SAFE)
    # ==================================================
    def apply_schema_from_json(self, csvname: str, stage="after", logger=None) -> pd.DataFrame:
        JSON_DIR.mkdir(exist_ok=True)
        base = Path(csvname).stem
        schema_path = JSON_DIR / f"{base}_schema_after.json"

        with open(schema_path, "r") as f:
            schema = json.load(f)

        for col, dtype in schema.items():
            if col not in self.df.columns:
                continue

            try:
                if dtype == "integer":
                    num = pd.to_numeric(self.df[col], errors="coerce")
                    values = num.dropna().to_numpy(dtype="float64", copy=False)

                    integer_safe = (
                        len(values) == 0 or
                        (np.isfinite(values).all() and (values == np.floor(values)).all())
                    )

                    if integer_safe:
                        self.df[col] = num.astype("Int64")
                    else:
                        self.df[col] = num.astype("float64")

                elif dtype == "float":
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

                elif dtype == "boolean":
                    self.df[col] = self.df[col].astype("boolean")

                elif dtype == "datetime":
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

                elif dtype == "category":
                    self.df[col] = self.df[col].astype("category")

                else:
                    self.df[col] = self.df[col].astype("string")

            except Exception as e:
                if logger:
                    logger.warning(f"[{csvname}] Skipped schema apply for {col}: {e}")

        return self.df
