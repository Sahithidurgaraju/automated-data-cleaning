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
# from src.pandasdatacleaning import Datacleaner
from sqlalchemy import create_engine,text
from src.config import DATA_DIR, JSON_DIR,CLEANED_DATA_DIR,PLOTS_DIR,VALIDATION_DIR,DATABASE_DIR,CLEANED_TRANFORM_DATA_DIR
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Datatranformer:
    def __init__(self, df,csvname):
        self.df = df.copy()
        self.csvname = csvname

    def plot_all_bins(self, df_with_bins, csvname, logger=None):
        if df_with_bins is None or df_with_bins.empty:
            if logger:
                logger.warning(f"[{csvname}] No data for bin plotting")
            return None
        bin_cols = [c for c in df_with_bins.columns if c.endswith("_bin")]
        if not bin_cols:
            if logger:
                logger.warning(f"[{csvname}] No bin columns found for plotting")
            return None
        PLOTS_DIR.mkdir(exist_ok=True)  
        clean_name = re.sub(r"_\d{8}_\d{6}$", "", Path(csvname).stem)
        csv_stem = clean_name
        folder = PLOTS_DIR / f"{csv_stem}_bins_histogram"
        folder.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = folder / f"{csv_stem}_bins_{timestamp}.png"
    # ---- Dynamic figure size (scales based on bin count) ----
        num_bins = len(bin_cols)
        fig_width = max(12, num_bins * 2.2)   # grows width if many bins
        fig_height = max(6, num_bins * 1.1)   # grows height if many bins
        plt.figure(figsize=(fig_width, fig_height))

    # ---- Plot each bin column ----
        for i, col in enumerate(bin_cols, 1):
            counts = df_with_bins[col].value_counts().sort_index()

            plt.subplot(num_bins, 1, i)
            plt.bar(counts.index.astype(str), counts.values)
            plt.title(f"{col} distribution")
            plt.xlabel(col)
            plt.ylabel("Count")

        plt.tight_layout()

    # ---- Save plot ----
        
        plt.savefig(file_path)
        plt.close()

        if logger:
            logger.info(f"[{csvname}] Bin histogram saved : {file_path}")

        return file_path

    def push_to_sql(self,df,logger,csvname):
        with open(DATABASE_DIR / f"sql_credentials.json", "r") as f:
            database = json.load(f)
        if not database.get("user") or not database.get("password") or not database.get("host"):
            logger.error("SQL credentials missing or blank. Please update sql_credentials.json")
            return None, None, None 
        try:

            engine = create_engine(f"mysql+pymysql://{database['user']}:{database['password']}@{database['host']}:{database['port']}", connect_args={
        "ssl": {
            "ca": "certs/isrgrootx1.pem"
        }
    },
    pool_pre_ping=True)

            engine = create_engine(f"mysql+pymysql://{database['user']}:{database['password']}@{database['host']}:{database['port']}")

        # Test connection
            with engine.connect() as conn:
                logger.info("MySQL connection successful!")

        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            return None, None, None 
        
        clean_name = re.sub(r"_\d{8}_\d{6}$", "", Path(csvname).stem)
        clean_name = clean_name.replace(" ", "_")
        dbname = database["dbname"]
        # dbname = clean_name
        table = f"transformed_{clean_name}".lower() 
        # with engine.connect() as conn:
        #     conn.execute(text(f"DROP DATABASE IF EXISTS `{dbname}`;"))
        #     conn.execute(text("FLUSH PRIVILEGES;"))
        #     logger.info(f"[{csvname}] Dropped database if existed: {dbname}")

        #  CREATE DATABASE fresh
    
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{dbname}`"))
            conn.execute(text("FLUSH PRIVILEGES;"))
            logger.info(f"[{csvname}] Created fresh MySQL database: {dbname}")
            
        #  Create new DB engine pointing to the fresh DB
        db_engine = create_engine(

            f"mysql+pymysql://{database['user']}:{database['password']}@{database['host']}:{database['port']}/{dbname}", connect_args={
        "ssl": {
            "ca": "certs/isrgrootx1.pem"
        }
    },
    pool_pre_ping=True)
        with db_engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

        bin_cols = [c for c in df.columns if c.endswith("_bin")]
        if bin_cols:
            df = df.drop(columns=bin_cols)
        ml_cols = [c for c in df.columns if c.endswith(("_scaled", "_log", "_outlier"))]
        df = df.drop(columns=ml_cols, errors="ignore")
        unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        df.to_sql(table, db_engine, if_exists="replace", index=False, chunksize=5000)
        if logger:
            logger.info(f"[{csvname}] Data pushed to DB `{dbname}` → table `{table}`")

        return dbname, table, db_engine
    

    def test_sql_data(self,dbname, table, db_engine, logger):
    # Check NULL count (excluding only year_end)
        schema_df = pd.read_sql(f"SELECT * FROM `{table}` LIMIT 0;", db_engine)
        cols = [c for c in schema_df.columns if c.lower() != "year_end"]

        null_query = "SELECT " + ", ".join([f"SUM(`{c}` IS NULL) AS `{c}_null_count`" for c in cols]) + f" FROM `{table}`;"
        null_result = pd.read_sql(null_query, db_engine)

    # Convert result to a readable dict
        null_counts = schema_df.isna().sum()

    # Fail test if any column (except year_end) has NULLs
        total_nulls = null_counts.drop(labels=[c for c in null_counts.index if c.lower() == "year_end"]).sum()
        assert total_nulls == 0, f"Unexpected NULLs found in table `{table}` inside DB `{dbname}` → {null_counts}"

    #  Preview data for sanity check (only 5 rows)
        preview = pd.read_sql(f"SELECT * FROM `{table}` LIMIT 5;", db_engine)
        assert not preview.empty, f"Table `{table}` is empty!"
    #  row count checking after pushed into sql database
        row_count = pd.read_sql(text(f"SELECT COUNT(*) AS row_count FROM `{table}`;"), db_engine)["row_count"][0]
        logger.info(f"Table `{table}` contains {row_count} rows")

        assert row_count > 0, f"No rows inserted into `{table}`!"
        logger.info(f"\nData Preview:\n{preview}")
        logger.info(f"\nNULL Counts:\n{null_counts}")
        logger.info(f"\nRow Counts:\n{row_count}")
        return int(null_counts.sum()), int(row_count)
    
    def generate_transform_config(self, df, csvname, logger):
        self.df = df
        config = {
        "dataset": csvname,
        "columns": {}
    }

        for col in df.columns:
            col_cfg = {"enabled": {}}

        # TEXT COLUMNS
            if df[col].dtype == "object":
                col_cfg["enabled"]["lowercase"] = True
                col_cfg["enabled"]["strip"] = True
                col_cfg["enabled"]["normalize_delimiters"] = (
                df[col].astype(str).str.contains(r"[;:/]").any()
                and not df[col].astype(str).str.contains(r"https?://").any()
            )

            # YEAR RANGE SPLIT DETECTION
                if df[col].astype(str).str.contains(r"https?://").any():
                    col_cfg["enabled"]["split_year"] = False
                    logger.info(f"[Skip] '{col}' is a URL column → skipping year split")
                else:
    # Run year range detection only if NOT a URL
                    s = df[col].astype("string")
                    s = s.str.replace("~", "-", regex=False)
                    s = s.str.replace(r"[–—−]", "-", regex=True)
                    s = s.str.replace(r"\s*-\s*", "-", regex=True)

                    years = s.str.findall(r"\d{4}")
                    has_range = years.str.len().gt(1).mean() > 0.1 and s.str.contains("-").any()

                    if has_range:
                        col_cfg["enabled"]["split_year"] = True
                        logger.info(f"[Detect] Year range detected in '{col}' → enabling split")
                    else:
                        col_cfg["enabled"]["split_year"] = False
                        logger.info(f"[Skip] '{col}' has no year range → skipping split")
        # NUMERIC COLUMNS
            if pd.api.types.is_numeric_dtype(df[col]):
                if col.lower().startswith("year") or col.lower().startswith("year_"):
                    col_cfg["enabled"]["cast_int"] = True
                else:
                    col_cfg["enabled"]["cast_float"] = True

                unique_ratio = df[col].nunique() / max(len(df), 1)
                col_cfg["enabled"]["bins"] = unique_ratio > 0.2

                col_cfg["enabled"]["scale"] = df[col].max() > 1000 or df[col].std() > 100

                from scipy.stats import skew
                col_cfg["enabled"]["log"] = df[col].dropna().min() > 0 and abs(skew(df[col].dropna())) > 1

                col_cfg["enabled"]["outliers"] = df[col].std() > 3

        # DATE DETECTION
            if df[col].astype(str).str.contains(r"\d{4}-\d{2}-\d{2}").mean() > 0.3:
                col_cfg["enabled"]["cast_datetime"] = True
                col_cfg["enabled"]["extract_date_parts"] = True

        # BOOLEAN DETECTION
            if set(df[col].dropna().astype(str).str.lower().unique()) <= {"true","false","yes","no","1","0"}:
                col_cfg["enabled"]["cast_bool"] = True

            config["columns"][col] = col_cfg

        return config

    def make_json_safe(self,obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.make_json_safe(v) for v in obj]
        return obj

    def save_config_file(self, df, csvname, logger):
        JSON_DIR.mkdir(exist_ok=True)
        clean_name = re.sub(r"_\d{8}_\d{6}$", "", Path(csvname).stem)
        base = clean_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schema_path = JSON_DIR / f"{base}_transformation_{timestamp}.json"

        for old_file in JSON_DIR.glob(f"{base}_transformation_*.json"):
            try:
                logger.info(f"Deleting: {old_file.name}")
                old_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {old_file}: {e}")

        config = self.generate_transform_config(df, csvname, logger)
        config = self.make_json_safe(config)
        with open(schema_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Transformation config saved: {schema_path.name}")
        return schema_path


    def apply_transformations(self, df, csvname, logger):
        df = df.copy()
        clean_name = re.sub(r"_\d{8}_\d{6}$", "", Path(csvname).stem)
        base = clean_name
        files = sorted(JSON_DIR.glob(f"{base}_transformation_*.json"), reverse=True)

        if not files:
            logger.warning(f"[{csvname}] No transformation config found")
            return df

        config_path = files[0]
        with open(config_path, "r") as f:
            config = json.load(f)

        logger.info(f"[{csvname}] Loaded transform config: {config_path.name}")

        for col, cfg in config["columns"].items():
            enabled = cfg["enabled"]

        # CASTING
            if enabled.get("cast_int"):
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

            if enabled.get("cast_float"):
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

            if enabled.get("cast_datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")

            if enabled.get("cast_bool"):
                df[col] = df[col].astype(str).str.lower().map({"true":True,"false":False,"yes":True,"no":False,"1":True,"0":False})

        # TEXT CLEANING
            if enabled.get("lowercase"):
                df[col] = df[col].astype(str).str.lower()

            if enabled.get("strip"):
                df[col] = df[col].astype(str).str.strip()

            if enabled.get("normalize_delimiters"):
                df[col] = df[col].astype(str).str.replace(r"[;:/]", ",", regex=True)

        # YEAR RANGE SPLIT
            if enabled.get("split_year")and not df[col].astype(str).str.contains("https?://").any():
                s = df[col].astype("string")
                s = s.str.replace("~", "-", regex=False)
                s = s.str.replace(r"[–—−]", "-", regex=True)
                s = s.str.replace(r"\s*-\s*", "-", regex=True)

    # assign cleaned string back before split
                df[col] = s  
                parts = df[col].astype(str).str.split("-", n=1, expand=True)
                if parts.shape[1] == 2:
                    df[f"{col}_start"] = pd.to_numeric(parts[0], errors="coerce", downcast="integer")
                    df[f"{col}_end"]   = pd.to_numeric(parts[1], errors="coerce", downcast="integer")
                    logger.info(f"[Transform] {col} split into {col}_start and {col}_end")
                else:
                    df[f"{col}_start"] = pd.to_numeric(parts[0], errors="coerce", downcast="integer")
                    df[f"{col}_end"]   = pd.Series([None] * len(df), dtype="Int64")  # keep end empty safely
                    logger.warning(f"[Skip] {col} had no valid range separator → only start extracted")
                
        # BINNING
            if enabled.get("bins") and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[f"{col}_bin"] = pd.qcut(df[col], 4, duplicates="drop").astype("category")
                except Exception as e:
                    logger.warning(f"[{csvname}] Binning failed for {col}: {e}")

        # SCALING
            if enabled.get("scale") and pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_scaled"] = (df[col] - df[col].mean()) / df[col].std()

        # LOG TRANSFORM
            if enabled.get("log") and pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_log"] = np.log1p(df[col])

        # OUTLIER HANDLING
            if enabled.get("outliers") and pd.api.types.is_numeric_dtype(df[col]):
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df[f"{col}_outlier"] = ~df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # DATE PART EXTRACTION
            if enabled.get("extract_date_parts") and df[col].dtype == "datetime64[ns]":
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day

    # PRINT TRANSFORMED COLUMNS
        logger.info("\n=== Columns after Transformation ===")
        for c in df.columns:
            logger.info(f"- {c}  |  dtype: {df[c].dtype}")
        savecsv = self.export_to_csv(logger, csvname=csvname,df_to_export=df)
        return df
    def export_to_csv(self,logger,csvname=None,df_to_export=None):
        CLEANED_TRANFORM_DATA_DIR.mkdir(exist_ok=True)
        df_to_export = df_to_export if df_to_export is not None else self.df
        csvname = Path(csvname).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if csvname:           
            filename = f"{csvname}_transformed_{timestamp}.csv"
        else:
            filename = f"cleaned_{timestamp}.csv"

        file_output_dir = CLEANED_TRANFORM_DATA_DIR / filename
        for old_file in CLEANED_TRANFORM_DATA_DIR.glob(f"{csvname}_*.csv"):
            try:
                print("Deleting:", old_file.name)
                old_file.unlink()
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")
        df_to_export.to_csv(file_output_dir, index=False, encoding="utf-8")
        logger.info(f"Data Exported to csv:{file_output_dir}")
        return file_output_dir
