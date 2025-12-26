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
from sqlalchemy import create_engine,text
from src.config import DATA_DIR, JSON_DIR,CLEANED_DATA_DIR,PLOTS_DIR,VALIDATION_DIR,DATABASE_DIR
# Ignore only RuntimeWarning (common for all-NaN median/mean)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Datatranformer:
    def __init__(self, df,csvname):
        self.df = df.copy()
        self.csvname = csvname

    def generate_transform_config(self, df, csvname,logger):
        self.df = df
        config = {
        "dataset": csvname,
        "columns": {},
        "dataset_ops": {
            "filter": {},
            "groupby": {
                "by":[],
                "agg":{}
            }
        }
    }

        for col in df.columns:
            col_cfg = {
                "suggested": {},
                "enabled": {}
            }

        # ---------- TEXT COLUMNS ----------
            if pd.api.types.is_string_dtype(df[col]):
                col_cfg["suggested"]["lowercase"] = {
                "reason": "Text column detected",
                "default": True
            }
                col_cfg["enabled"]["lowercase"] = True

                col_cfg["suggested"]["strip"] = {
                "reason": "Leading/trailing spaces possible",
                "default": True
            }
                col_cfg["enabled"]["strip"] = True

            # Detect delimiter-heavy text (skills, tags, categories)
                if df[col].astype(str).str.contains(r"[;/,:]").any():
                    col_cfg["suggested"]["normalize_delimiters"] = {
                    "reason": "Multiple delimiters detected (; / : ,)",
                    "default": True
                }
                    col_cfg["enabled"]["normalize_delimiters"] = True

                    col_cfg["suggested"]["split_multi_values"] = {
                    "reason": "Multi-valued text detected (skills/tags)",
                    "default": False
                }
                    col_cfg["enabled"]["split_multi_values"] = True

        # ---------- NUMERIC COLUMNS ----------
            if pd.api.types.is_numeric_dtype(df[col]):
    # Special handling for year columns (no float suggestion)
                if col.lower().startswith("year") or col.lower().startswith("year_"):
                    col_cfg["suggested"]["cast"] = {
            "type": "int",
            "reason": "Year column should remain integer",
            "default": True
        }
                    col_cfg["enabled"]["cast"] = True  # still enabled, but will cast to int later
                else:
                    col_cfg["suggested"]["cast"] = {
            "type": "float",
            "reason": "Numeric column",
            "default": True
        }
                    col_cfg["enabled"]["cast"] = True

                unique_ratio = df[col].nunique() / max(len(df), 1)
                if unique_ratio > 0.2:
                    col_cfg["suggested"]["bins"] = {
                    "type": "auto",
                    "reason": "High cardinality numeric column",
                    "default": False
                }
                    col_cfg["enabled"]["bins"] = False

                config["dataset_ops"]["filter"][col] = {
                ">=": None,
                "<=": None
            }

            # suggest aggregation
        # ---------- YEAR SPLIT ----------
            if (
            "year" in col.lower()
            and df[col].dtype == "object"
            and df[col].astype(str).str.contains("-").any()
        ):
                col_cfg["suggested"]["split_year"] = {
                "reason": "Year range detected (e.g. 2018-2020)",
                "default": True
            }
                col_cfg["enabled"]["split_year"] = True

            if col_cfg["suggested"]:
                config["columns"][col] = col_cfg

        return config
        
    def save_config_file(self,df,csvname,logger):
        JSON_DIR.mkdir(exist_ok=True)
        base = Path(csvname).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schema_path = JSON_DIR / f"{base}_transformation_{timestamp}.json"
        for old_file in JSON_DIR.glob(f"{base}_transformation_*.json"):
            try:
                print("Deleting:", old_file.name)
                old_file.unlink()
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")
        config = self.generate_transform_config(df, csvname,logger)
    # Save schema
        with open(schema_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Transformation config saved: {schema_path}")
        return schema_path


    def apply_transformations(self, df, csvname, logger):
        df = df.copy()

        base = Path(csvname).stem

        files = sorted(JSON_DIR.glob(f"{base}_transformation_*.json"), reverse=True)
        if not files:
            logger.warning(f"[{csvname}] No transformation config found")
            return df, {}

        config_path = files[0]
        logger.info(f"[{csvname}] Using transform config: {config_path.name}")

        with open(config_path, "r") as f:
            config = json.load(f)

        execution_report = {
            "dataset": csvname,
            "applied_at": datetime.now().isoformat(),
            "column_transformations": {},
            "dataset_operations": {},
            "generated_columns": []
        }

        for col, cfg in config.get("columns", {}).items():
            if col not in df.columns:
                continue

            enabled = cfg.get("enabled", {})
            col_report = []

            # CAST
            if enabled.get("cast"):
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
                col_report.append("cast_to_numeric")

            # LOWERCASE
            if enabled.get("lowercase"):
                df[col] = df[col].str.lower()
                col_report.append("lowercase")

            # STRIP
            if enabled.get("strip"):
                df[col] = df[col].str.strip()
                col_report.append("strip")

            # NORMALIZE DELIMITERS
            if enabled.get("normalize_delimiters"):
                df[col] = df[col].str.replace(r"[;:/]", ",", regex=True).str.replace(r"\s*,\s*", ",", regex=True)
                col_report.append("normalize_delimiters")

            # EXPLODE
            if enabled.get("split_multi_values"):
                df[col] = df[col].str.lower().str.strip().str.replace(";", ",", n=-1)

            # SPLIT YEAR RANGE
            if enabled.get("split_year"):
                parts = df[col].str.split("-", n=1, expand=True)
                df[f"{col}_start"] = pd.to_numeric(parts[0], errors="coerce", downcast="integer")
                df[f"{col}_end"] = pd.to_numeric(parts[1], errors="coerce", downcast="integer")
                col_report.append("split_year")

            # BINNING
            if enabled.get("bins") is True and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    bin_col = f"{col}_bin"
                    df[bin_col] = pd.qcut(df[col], q=4, duplicates="drop").astype("category")
                    execution_report["generated_columns"].append(bin_col)
                    execution_report["generated_columns"].append(bin_col)
                    col_report.append("bins_created")
                    df_with_bins = df.copy()
                except Exception as e:
                    logger.warning(f"[{csvname}] BINNING failed: {e}")

            if col_report:
                execution_report["column_transformations"][col] = col_report

        # FILTER
        filter_cfg = config.get("dataset_ops", {}).get("filter", {})
        applied_filters = []
        for col, conds in filter_cfg.items():
            if col not in df.columns or not isinstance(conds, dict):
                continue
            for op, val in conds.items():
                if val is None:
                    continue
                if op == ">=":
                    df = df[df[col] >= val]
                elif op == "<=":
                    df = df[df[col] <= val]
                elif op == "==":
                    df = df[df[col] == val]
                applied_filters.append(f"{col} {op} {val}")

        if applied_filters:
            execution_report["dataset_operations"]["filters"] = applied_filters

        # GROUPBY
        groupby_cfg = config.get("dataset_ops", {}).get("groupby", {})
        by = groupby_cfg.get("by", [])
        agg = groupby_cfg.get("agg", {})
        if by and agg:
            df = df.groupby(by, as_index=False).agg(agg)

        # SAVE EXECUTION REPORT
        report_path = JSON_DIR / f"{base}_transformation_execution.json"
        with open(report_path, "w") as f:
            json.dump(execution_report, f, indent=2)
        return df, execution_report

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
        csv_stem = Path(csvname).stem.lower().replace(" ", "_")
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

    
    def get_database_credentials(self,logger):
        DATABASE_DIR.mkdir(exist_ok=True)
        database={
            "user":"root",
            "password":"root",
            "localhost":"127.0.0.1",
            "port":3306
        }
        report_path = DATABASE_DIR / f"sql_credentials.json"
        # report_path.mkdir(exist_ok=True)
        with open(report_path,"w") as f:
            json.dump(database, f,indent=4)
        logger.info(f"{report_path} created")
        return report_path
    

    def push_to_sql(self,df,logger,report_path,csvname):
        df = self.df
        with open(report_path, "r") as f:
            database = json.load(f)
        engine = create_engine(f"mysql+pymysql://{database['user']}:{database['password']}@{database['localhost']}:{database['port']}")
        dbname = csvname.lower().replace(".csv", "").replace(" ", "_")
        table = f"transformed_{csvname.lower().replace('.csv','')}"
        check_db_query = text(f"SHOW DATABASES LIKE '{dbname}';")        
        with engine.connect() as conn:
            exists = conn.execute(check_db_query).fetchone()
        # 2. Create DB if not exists
            if not exists:
                conn.execute(text(f"CREATE DATABASE `{dbname}`;"))
                conn.execute(text("FLUSH PRIVILEGES;"))
                if logger:
                    logger.info(f"[{csvname}] Created MySQL database: {dbname}")
        db_engine = create_engine(f"mysql+pymysql://{database['user']}:{database['password']}@{database['localhost']}:{database['port']}/{dbname}")
        try:
            with db_engine.connect() as conn:
                logger.info(f"{db_engine} MySQL connection successful!")
        except Exception as e:
            logger.info(" Connection failed:", e)
        
        df.to_sql(table, db_engine, if_exists="replace", index=False, chunksize=5000)
        if logger:
            logger.info(f"[{csvname}] Data pushed to DB `{dbname}` → table `{table}`")

        return dbname, table, db_engine
    

    def test_sql_data(self,dbname, table, db_engine, logger):
    # 1. Check NULL count (excluding only year_end)
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