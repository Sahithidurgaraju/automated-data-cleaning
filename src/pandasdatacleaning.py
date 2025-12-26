import pandas as pd
import warnings
import numpy as np
import re,os,unicodedata,time,glob,json,seaborn as sns, matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from src.config import DATA_DIR, JSON_DIR,CLEANED_DATA_DIR,PLOTS_DIR,VALIDATION_DIR
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#pattern checking 
TEXT_DIRTY_PATTERN = r"\[[^\]]*\]|[†‡*]|[\x00-\x1F\x7F]"
NUMERIC_PATTERN = r"[\$\€\£\₹(),]|\d+[KMBkmb]$"
YEAR_PATTERN = r"(19\d{2}|20\d{2})"
YEAR_RANGE_PATTERN = r"(19\d{2}|20\d{2}).*(19\d{2}|20\d{2})"
#cleaning threshold
TEXT_THRESHOLD = 0.6
YEAR_THRESHOLD = 0.6
NUMERIC_THRESHOLD = 0.1
#for checking the id columns
ID_NAME_KEYWORDS = (
    "id", "uuid", "guid", "hash", "code", "ref", "reference"
)

#datacleaner class starts 
class Datacleaner:
    def __init__(self, df,csvname):
        self.df = df.copy()
        self.logs = []
        self.column_mapping = {}  
        self.csvname = csvname

    #cleaning column names
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

    #Load data from csv
    def load_csv(self, csv_path, logger=None, encoding="utf-8",errors = "replace"):
        """
        Load CSV safely, clean column names, clean text columns efficiently.
        """
        # Normalize path
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
        # Read CSV (encoding fallback)
        try:
            self.df = pd.read_csv(csv_path, na_values=[""],
            keep_default_na=True,encoding=encoding)
            if logger:
                logger.info(f"Loaded CSV: {csv_path} ({encoding})")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="latin1")
            if logger:
                logger.warning(f"Encoding fallback to latin1: {csv_path}")
        # Clean column names
        self.clean_colnames()

        if logger:
            logger.info(f"Cleaned column names: {list(self.df.columns)}")

        # Clean TEXT columns using vectorization
        
        text_cols = self.df.select_dtypes(include=["object", "string"]).columns

        for col in text_cols:
            s = self.df[col]

    # unicode fixes (vectorized)
            s = (
            s.astype("string")
         .str.replace("â€ ", "", regex=False)
         .str.replace("â€¡", "", regex=False)
         .str.replace("Ã©", "é", regex=False)
    )

    # APPLY ONCE 
            self.df[col] = self.clean_text(s)

            if logger:
                logger.info(f"Cleaned text column: {col}")

        if logger:
            logger.info(f"CSV loading and cleaning completed: {csv_path}")

        return self.df
    def shape(self, csvname,logger):
        if self.df is None:
            logger.warning(f"[{csvname}] No dataframe loaded")
            return None
        shape = self.df.shape
        rows = shape[0]
        columns = shape[1]
        self.logs.append(shape)
        logger.info(f"shape:{shape}")
        logger.info(f"Rows present in dataframe: {rows}")
        logger.info(f"Columns present in dataframe: {columns}")
        return shape,rows,columns

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
        self.df = df
        return self.df
    def is_year_column(self, s: pd.Series) -> bool:
        s = s.dropna()
        if s.empty:
            return False
        if not pd.api.types.is_numeric_dtype(s):
            return False
        return s.between(1800, 2200).mean() >= 0.8

    def handle_missing(self, csvname, logger, strategy="auto", fill_value=None):
        """
        Handles missing values using median and mode. Options include Auto, Drop, or Fill with a custom value.
        """

        if self.df is None:
            logger.warning(f"[{csvname}] No dataframe loaded")
            return None
        df = self.df
        
        df = self.normalize_missing(self.df.copy())
            
    # DROP STRATEGY
    
        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(f"[{csvname}] Dropped {before - len(df)} rows with missing values")
            self.df = df
            return self.df
            
    # FILL STRATEGY
    
        if strategy == "fill":
            if fill_value is None:
                logger.warning(f"[{csvname}] fill_value required for fill strategy")
                return self.df

            self.df = df.fillna(fill_value)
            logger.info(f"[{csvname}] Filled all missing values with {fill_value}")
            return self.df
    
    # AUTO STRATEGY 
    
        if strategy == "auto":

        # Drop fully empty columns
            before_cols = df.shape[1]
            df = df.dropna(axis=1, how="all")
            dropped_cols = before_cols - df.shape[1]

            if dropped_cols > 0:
                logger.info(f"[{csvname}] Dropped {dropped_cols} fully empty columns")
            
            for col in df.columns:
                if df[col].dtype in ["object", "string"]:
                    # mode = df[col].mode()
                    # fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    # df[col] = df[col].fillna(fill_val)
                    numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
                    if numeric_ratio >= 0.6:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Split columns by dtype ONCE
            num_cols = df.select_dtypes(include=["number"]).columns
            dt_cols = df.select_dtypes(include=["datetime"]).columns
            obj_cols = df.select_dtypes(include=["object", "string"]).columns
            cat_cols = df.select_dtypes(include=["category"]).columns
        # Numeric → median (vectorized)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
            for col in num_cols:
                    s = df[col]
                    if self.is_year_column(df[col]):
                        continue
                    if s.isna().any():
                        mode = s.mode()
                        if not mode.empty:
                            fill_val = mode.iloc[0]
                        else:
                            fill_val = s.median()

                        if pd.notna(fill_val):
                            df[col] = s.fillna(fill_val)
            
        #  Datetime → mode or min
            for col in dt_cols:
                if df[col].isna().any():
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else pd.Timestamp.min
                    df[col] = df[col].fillna(fill_val)

        # Text → mode 
            for col in obj_cols:
                if df[col].isna().any():
                    # df.replace("", np.nan, inplace=True)
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    df[col] = df[col].fillna(fill_val)

            for col in cat_cols:
                if df[col].isna().any():
                    # df.replace("", np.nan, inplace=True)
                    mode = df[col].mode()
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    df[col] = df[col].fillna(fill_val)

            logger.info(
                f"[{csvname}] Missing values handled | "
                f"numeric={len(num_cols)}, "
                f"text={len(obj_cols) + len(cat_cols)}, "
                f"datetime={len(dt_cols)}"
        )
            self.df = df
            return self.df

        logger.warning(f"[{csvname}] Unknown missing value strategy: {strategy}")
        return self.df

    def load_schema(self,csvname):
        """
        Loads the schema configuration from a JSON file for the given CSV file.
        """
        JSON_DIR.mkdir(exist_ok=True)
        base = os.path.splitext(os.path.basename(csvname))[0]
        return json.load(open(JSON_DIR / f"{base}_schema_after.json"))


    def get_outlier_columns(self, df, schema):
        """
        Identifies columns in the DataFrame that contain outliers.
        """
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
    """

        if when == "before":
        # Detect numeric-like columns dynamically
            outlier_cols = []
            for col in self.df.columns:
                s = self.df[col]
                if any(k in col.lower() for k in ["year", "yr"]):
                    continue

            # numeric
                if pd.api.types.is_numeric_dtype(s):
                    if pd.api.types.is_bool_dtype(s):
                        continue
                    s_nonnull = s.dropna()
                    if s_nonnull.empty or s_nonnull.nunique() <= 1:
                        continue
                    outlier_cols.append(col)
                    continue
                

            # numeric-like strings
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


    # Outlier clipping for visualization 
        for col in outlier_cols:
            Q1 = plot_df[col].quantile(0.25)
            Q3 = plot_df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0 or pd.isna(IQR):
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            plot_df[col] = plot_df[col].clip(lower, upper)

    # Output directory 
        PLOTS_DIR.mkdir(exist_ok=True)
        output_dir = PLOTS_DIR / f"{csvname}_outliers"
        output_dir.mkdir(exist_ok=True)

    # Cleanup 
        if cleanup_old:
            for old in output_dir.glob(f"{csvname}_outliers_{when}_*.png"):
                try:
                    old.unlink()
                except Exception:
                    pass

    #  Plot 
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

    def is_structural_na_column(self, col: str, s: pd.Series) -> bool:
        col_lower = col.lower()

    # year_end, end_year, *_end
        if col_lower.endswith("_end"):
            return True

    # numeric year-like columns
        if pd.api.types.is_numeric_dtype(s):
            s_non_null = s.dropna()
            if not s_non_null.empty:
                if s_non_null.between(1800, 2200).mean() >= 0.8:
                    return True

        return False


    def plot_missing_values(self,csvname, df, logger, show_plot=False,when="before",cleanup_old=False):
        PLOTS_DIR.mkdir(exist_ok=True)
        csvname = Path(csvname).stem
        print(self.logs.append(csvname))
        plot_cols = [col for col in df.columns 
                     if not self.is_structural_na_column(col, df[col])]
        missing_values = df[plot_cols].isna().sum()
        missing_values = missing_values[missing_values > 0]
        logger.info(f"missing values found:{missing_values}")
        self.logs.append(f"Missing values found: {missing_values}")        
        if missing_values.empty:
            logger.info("No missing values found")
            self.logs.append(f"No missing values found in {csvname}.")
            missing_values = pd.Series([0]*len(self.df.columns), index=self.df.columns)
            #plotting the bar graph of missing values present in the data
        plt.figure(figsize=(8,12))
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
        timestamp = time.strftime("%Y%m%d_%H%M")  

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

    def dirty_ratio(self, s: pd.Series, pattern: str) -> float:
        s = s.dropna().astype(str)
        if s.empty:
            return 0.0
        return s.str.contains(pattern, regex=True).mean()

    def is_date_column(self, s: pd.Series) -> bool:
        s = s.dropna().astype(str)
        if s.empty:
            return False
        return pd.to_datetime(s, errors="coerce").notna().mean() >= 0.6

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

    def extract_year(self, s: pd.Series):
        s = s.astype(str)
        s = s.str.replace(r"[–—−]", "-", regex=True)
        s = s.str.replace(r"(\d{4})(?=\d{4})", r"\1-", regex=True)

        years = s.str.findall(YEAR_PATTERN)
        start = years.str[0].astype("Int64")
        end = years.str[-1].where(years.str.len() > 1).astype("Int64")

        return start, end

    def standard_data(self, csvname=None, logger=None) -> pd.DataFrame:
        cleaned_cols, skipped_cols = [], []

        for col in self.df.columns:
            s = self.df[col]

            # ID PROTECTION 
            if self.is_id_column(col,s):
                self.df[col] = s.astype("string").str.strip()
                skipped_cols.append(col)
                continue

            #  DIRTY RATIOS 
            text_ratio = self.dirty_ratio(s, TEXT_DIRTY_PATTERN)
            numeric_ratio = self.dirty_ratio(s, NUMERIC_PATTERN)
            year_range_ratio = self.dirty_ratio(s, YEAR_RANGE_PATTERN)
            year_presence_ratio = self.dirty_ratio(s, YEAR_PATTERN)

            # DATE 
            if self.is_date_column(s) and text_ratio >= TEXT_THRESHOLD:
                self.df[col] = pd.to_datetime(self.clean_text(s), errors="coerce")
                cleaned_cols.append(col)
                continue

            # YEAR 
            if year_presence_ratio >= YEAR_THRESHOLD and year_range_ratio >= YEAR_THRESHOLD:                
                y1, y2 = self.extract_year(s)
                self.df[col] = y1
                if y2.notna().any():
                    self.df[f"{col}_end"] = y2
                cleaned_cols.append(col)
                continue

            #  NUMERIC 
            if numeric_ratio >= NUMERIC_THRESHOLD:
                self.df[col] = self.clean_numeric(s)
                cleaned_cols.append(col)
                continue

            #TEXT 
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

    # purely numeric 
        pure_numeric = s.str.match(r"^\d+(\.\d+)?$", na=False).mean()

        return has_letters >= 0.6 and pure_numeric < 0.2
    
    def _detect_column_type(self, s: pd.Series) -> str:
        s_non_null = s.dropna()

        if s_non_null.empty:
            return "text"

        # BOOLEAN
        lowered = s_non_null.astype(str).str.lower().unique()
        if set(lowered).issubset({"0", "1", "true", "false", "yes", "no"}):
            return "boolean"

        # NUMERIC 
        num = pd.to_numeric(s_non_null, errors="coerce")
        numeric_ratio = num.notna().mean()

        if numeric_ratio >= 0.95:
            values = num.dropna().to_numpy(dtype="float64", copy=False)
            integer_ratio = (values == np.floor(values)).mean()

            if integer_ratio >= 0.99:
                return "integer"
            return "float"

        # DATETIME 
        dt = pd.to_datetime(s_non_null, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.95:
            return "datetime"

        #  CATEGORY 
        cardinality_ratio = s_non_null.astype(str).nunique() / len(s_non_null)
        if cardinality_ratio <= 0.1:
            return "category"

        # TEXT
        return "text"

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

    # Detect schema 
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
    
    def derive_duplicate_subset(self, csvname):
        df = self.df
        schema = self.load_schema(csvname)

    # Strong ID columns (highest priority)
        id_cols = [
        col for col in df.columns
        if self.is_id_column(col, df[col])
        and df[col].notna().mean() > 0.95   
    ]

        if id_cols:
            return id_cols

    # Business keys (controlled, not all text columns)
        business_candidates = []
        for col, dtype in schema.items():
            if col not in df.columns:
                continue

            if dtype in ("category", "object", "text"):
                uniqueness = df[col].nunique(dropna=True) / len(df)

            # only semi-unique columns
                if 0.3 < uniqueness < 0.98:
                    business_candidates.append(col)

            if "year" in col.lower():
                business_candidates.append(col)

    # deterministic order
        business_candidates = list(dict.fromkeys(business_candidates))

        return business_candidates[:3]

    def remove_duplicates(self, csvname, logger, subset=None, keep="first"):
        """
    Remove duplicate rows safely.

    subset:
        None            -> full row duplicates
        list[str]       -> key-based duplicates
    """
        self.deduplication_status = {
        "executed": False,
        "aborted": False,
        "keys": None,
        "removed": 0
    }
        if self.df is None:
            self.deduplication_status["reason"] = "No data loaded"
            logger.warning(f"[{csvname}] No dataframe loaded")
            return None

        before = len(self.df)
        if subset is None:
            self.deduplication_status["reason"] = "No valid deduplication keys"
            subset = self.derive_duplicate_subset(csvname)
        subset = [c for c in subset if c in self.df.columns]
        if not subset:
                logger.warning(
                f"[{csvname}] Duplicate subset empty after filtering — skipping deduplication"
            )
                return self.df
        data = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(data)
        self.deduplication_status["keys"] = subset
        self.deduplication_status["removed"] = removed
        logger.info(
                f"[{csvname}] Removed {removed} duplicate rows "
                f"using keys={subset}"
        )
        removal_ratio = removed / before
        if removal_ratio > 0.3:
            logger.error(
            f"[{csvname}] Aborting deduplication — "
            f"{removal_ratio:.1%} rows flagged as duplicates using {subset}"
        )
            self.deduplication_status.update({
            "executed": False,
            "aborted": True,
            "keys": subset,
            "removed": 0,
            "reason": f"Aborted: {removal_ratio:.1%} rows would be removed"
        })
            return self.df
        self.deduplication_status.update({
        "executed": True,
        "aborted": False,
        "keys": subset,
        "removed": removed
    })
        logger.info(
        f"[{csvname}] Removed {removed} duplicate rows using keys={subset}"
    )

        self.df = data
        return data
    
    def matches_pattern(col, pattern):
        return pattern.startswith("*") and col.endswith(pattern[1:])


    def validate_cleaned_data(self,csvname,logger,df,rows_before=None):
        """
    Run post-cleaning data validation and return results dict.
    """
        df = self.df
        results = {
        "dataset": csvname,
        "status": "PASS",
        "checks": {}
    }

        def record(name, passed, message=""):
            results["checks"][name] = {
            "status": "PASS" if passed else "FAIL",
            "message": message
        }
            if not passed:
                results["status"] = "FAIL"

    #  Not empty
        try:
            assert len(df) > 0
            record("non_empty", True,f"Dataset contains {len(df)} rows after cleaning")
        except AssertionError:
            record("non_empty", False, "Dataset empty after cleaning")

    #  Schema check
        try:
            schema = self.load_schema(csvname)
            expected = set(schema.keys())
            actual = set(df.columns)
            assert expected.issubset(actual)
            record("schema_match", True,"All expected schema columns are present")
        except AssertionError:
                record(
                "schema_match",
                False,
                f"Missing columns: {expected - actual}"
        )
    #  Deduplication check
        dedup_status = getattr(self, "deduplication_status", None)

        if not dedup_status:
            record(
        "deduplication",
        True,
        "Deduplication not executed"
    )

        elif not dedup_status.get("applied", False):
            record(
        "deduplication",
        True,
        f"Deduplication skipped intentionally: {dedup_status.get('reason')}"
    )

        else:
            dedup_keys = dedup_status.get("keys", [])
            dupes = df.duplicated(subset=dedup_keys).sum()

            if dupes == 0:
                record("deduplication", True)
            else:
                record(
            "deduplication",
            False,
            f"{dupes} duplicate rows remain for keys={dedup_keys}"
        )
        #null values 
        allowed_null_patterns = ["*_end"]

        total_nulls = 0
        structural_nulls = 0

        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue

            total_nulls += null_count

            is_structural = (
            hasattr(self, "is_structural_na_column") and
            self.is_structural_na_column(col, df[col])
        ) or any(self.matches_pattern(col, p) for p in allowed_null_patterns)

            if is_structural:
                structural_nulls += null_count
            else:
                record(
                f"unexpected_nulls_{col}",
                False,
                f"Found {null_count} unexpected null values"
            )

        if total_nulls == 0:
            record(
            "no_nulls",
            True,
            f"No null values found in this {csvname} dataset"
        )
        elif total_nulls == structural_nulls:
            record(
            "no_nulls",
            True,
            f"Only structural nulls found ({structural_nulls}); expected"
        )
        else:
            record(
            "no_nulls",
            False,
            f"Found {total_nulls - structural_nulls} unexpected null values"
        )
    # 5️⃣ Row drop guard
        if rows_before:
            drop_ratio = (rows_before - len(df)) / rows_before
            passed = drop_ratio <= 0.3
            record(
            "row_drop_ratio",
                passed,
            f"Dropped {drop_ratio:.1%} of rows"
        )

        logger.info(
        f"[{csvname}] Validation completed with status={results['status']}"
    )

        return results


    def datacleaning_pipeline(self, csvname, logger,cleanup_old=False, strategy=None,show_plot=False):
    
        self.load_csv(csvname,logger)
        rows_before = len(self.df) 
        self.shape(csvname,logger)
        self.plot_outliers(csvname, logger,when="before", cleanup_old=True)
        self.plot_missing_values(csvname, self.df, logger, show_plot=False,when="before",cleanup_old=True)
        self.standard_data(csvname,logger)
        self.handle_missing(csvname,logger,strategy=strategy)
        self.detect_and_save_schema(df=self.df, csvname=csvname,stage="after", logger=logger)
        self.apply_schema_from_json(csvname=csvname,stage="after",logger=logger)
        self.remove_irrelevant_columns(csvname,logger)
        self.plot_outliers(csvname,logger,when="after", cleanup_old=True)
        self.remove_duplicates(csvname,logger)
        self.get_cleaned_data(csvname,logger)
        self.plot_missing_values(csvname, self.df, logger, when="after",show_plot=False)
        validation_results = self.validate_cleaned_data(csvname=csvname,logger=logger,df=self.df)
        self.save_validation_json(csvname,validation_results,logger)
        self.export_to_csv(csvname, df_to_export=self.df)
        return self.df
    
    #assertions to validate the cleaning data
    def save_validation_json(self, csvname, results, logger):
        VALIDATION_DIR.mkdir(exist_ok=True)
        results["timestamp"] = time.strftime("%Y%m%d_%H%M")

        path = VALIDATION_DIR/ f"{csvname}_validation.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"[{csvname}] Validation JSON saved to {path}")