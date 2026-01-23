import streamlit as st
import pandas as pd,numpy as np
from sqlalchemy import create_engine
import os,json,glob,pytest,re,requests
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from logger_config import get_logger 
from pandasdatacleaning import Datacleaner
import pyarrow as pa
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data_output"
DATA_DIR = PROJECT_ROOT/"data"
PLOTS_DIR = PROJECT_ROOT / "plots"
DASHBOARD_DIR =  PROJECT_ROOT/"dashboard_reports"

OWNER = "Sahithidurgaraju"
REPO = "automated-data-cleaning"
RELEASE_TAG = "latest-images"
#secrets
DB_HOST = st.secrets["tidb"]["host"]
DB_USER = st.secrets["tidb"]["user"]
DB_PASSWORD = st.secrets["tidb"]["password"]
DB_NAME = st.secrets["tidb"]["database"]
DB_PORT = st.secrets["tidb"]["port"]

@st.cache_data(ttl=300)
def get_release_assets():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{RELEASE_TAG}"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    return r.json().get("assets", [])

@st.cache_data(show_spinner=False)
def sql_query(dbname, table_name):
    credential_file = os.path.join(os.path.dirname(__file__), "..", "sql_credentials", "sql_credentials.json")
    with open(credential_file) as f:
        database = json.load(f)

    mysqluri = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

    if not DB_USER or not DB_PASSWORD or not DB_HOST:
        st.error("MySQL credentials are missing. Please update `sql_credentials.json` in the project root.")
        engine = None
    else:
        engine = create_engine(mysqluri,connect_args={
        "ssl": {
            "ca": "certs/isrgrootx1.pem"
        }
    },
    pool_pre_ping=True
)
    return pd.read_sql(f"SELECT * FROM `{dbname}`.`{table_name}`", engine)

@st.cache_data(show_spinner=False)
def rows_columns_count(df_before,df_after):
    rows_before = len(df_before)
    rows_after  = len(df_after)
    cols_before = len(df_before.columns)
    cols_after  = len(df_after.columns)
    return rows_before, rows_after,cols_before, cols_after

class Datadashboard:
    def __init__(self):
        credential_file = os.path.join(os.path.dirname(__file__), "..", "sql_credentials", "sql_credentials.json")
        with open(credential_file) as f:
            database = json.load(f)


        self.mysqluri = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
        if not DB_USER or not DB_PASSWORD or not DB_HOST:
            st.error("MySQL credentials are missing. Please update `sql_credentials.json` in the project root.")
            self.engine = None
        else:
            try:
                self.engine = create_engine(self.mysqluri,connect_args={
        "ssl": {
            "ca": "certs/isrgrootx1.pem"
        }
    },
    pool_pre_ping=True
)
        # Optional: test connection
                with self.engine.connect() as conn:
                    st.success("Connected to MySQL successfully!")
            except Exception as e:
                    st.error(f"MySQL connection failed: {e}")
                    self.engine = None

    def get_dataset_databases(self):
        # Fetch all databases and filter dataset DBs
        try:
            dbs = pd.read_sql("SHOW DATABASES", self.engine)
            dataset_dbs = [d for d in dbs["Database"] if d not in ("mysql","information_schema","performance_schema","sys")]

            return dataset_dbs
        except Exception as e:
            raise e
    def read_csv_with_fallback(self, file_path):
        file_path = Path(file_path)
        encodings = ["utf-8", "cp1252", "latin1"]


        for enc in encodings:
            try:
                df = pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=enc,
                na_values=["", "NA", "N/A", "None", "null", "-", "?", "Nan", "Inf", "Not Applicable"],
                keep_default_na=True,
                skip_blank_lines=True
            )

                # if logger:
                #     logger.info(f"Loaded CSV using encoding: {enc}")
            # basic sanity check
                if df.empty or df.shape[1] < 2:
                    raise ValueError("CSV appears invalid after parsing")

                return df

            except UnicodeDecodeError:
                continue

    # if all encodings fail
        st.error(
        "❌ Unable to read CSV file. "
        "The file may be corrupted or use an unsupported encoding."
    )

        # if logger:
        #     logger.info(f"CSV load failed")

        st.stop()

    # def get_tables_from_databases(self,dataset_dbs):
    #     # Collect all tables from dataset DBs
    #     tables = []
    #     try:
    #         for db in dataset_dbs:
    #             tbls = pd.read_sql(f"SHOW TABLES FROM `{db}`", self.engine)
    #             for t in tbls.iloc[:, 0]:
    #                 tables.append({"db": db, "table": t})
    #         return tables
    #     except Exception as e:

    #         raise e
    def get_tables_from_database(self, dbname):
        tables = []
        try:
            query = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{dbname}'
              AND table_type = 'BASE TABLE'
              AND LEFT(table_name, 12) = 'transformed_';
        """

            df = pd.read_sql(query, self.engine)

        # normalize column names
            df.columns = df.columns.str.lower().str.strip()

            for t in df["table_name"]:
                tables.append(t)

            return tables

        except Exception as e:
            raise e

    def find_images(self, assets, must_contain):
        results = []
        for a in assets:
            name = a["name"].lower()
            if all(key.lower() in name for key in must_contain):
                results.append(a["browser_download_url"])
        return results

    def find_dashboard_pdf(self, assets, cleaned_csv):
        matches = []
        csvname = cleaned_csv + "_dashboard"
        for asset in assets:
            name = asset["name"].lower()

            if (
                csvname.lower() in name
            and "cleaned_dashboard" in name
            and name.endswith(".pdf")
        ):
                matches.append(asset)

    # If multiple PDFs exist, return the latest one
        if matches:
            matches.sort(key=lambda x: x["name"], reverse=True)
            return matches[0]["browser_download_url"]

        return matches

    def load_table(self, dbname, table_name):
        return sql_query(dbname, table_name)
    
    def rows_columns_count(self, csvname,dbname,table_name):

        df_before = self.read_csv_with_fallback(DATA_DIR/csvname)
    
        df = self.load_table(dbname, table_name)
        df_after = pd.DataFrame(df)
        return rows_columns_count(df_before,df_after)

    def make_arrow_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
            # normalize everything to string first
                df[col] = df[col].astype(str)

            # remove common non-numeric placeholders
                df[col] = (
                df[col]
                .str.strip()
                .replace(
                    {
                        "-": np.nan,
                        "–": np.nan,
                        "—": np.nan,
                        "": np.nan,
                        "nan": np.nan,
                        "None": np.nan,
                    }
                )
            )

            # remove % sign if present
                if df[col].str.contains("%", na=False).any():
                    df[col] = df[col].str.replace("%", "", regex=False)

            # attempt numeric conversion
                df[col] = pd.to_numeric(df[col], errors="ignore")

        return df


    def cleaning_score(self,df_before,df_after,dedupe_status,csvname,dbname,table_name):
        """
    Returns a REAL cleaning score based on actual improvements.
    """


    # NULL IMPROVEMENT

        ignore_cols = [
            c for c in df_after.columns
            if c.endswith("_end") or c.endswith("_start") or "_bin" in c
    ]

        base_cols = [c for c in df_before.columns if c in df_after.columns and c not in ignore_cols]

        null_before = df_before.isna().mean().mean() * 100
        null_after  = df_after.isna().mean().mean() * 100
        null_improve = round(null_before - null_after, 2)


    # DELIMITER NOISE

        NOISE_PATTERN = r"[;:|/]"

        s_before = df_before.select_dtypes(include=["object","string"]).stack().dropna().astype("string")
        s_after  = df_after.select_dtypes(include=["object","string"]).stack().dropna().astype("string")

    # exclude URLs
        s_before = s_before[~s_before.str.contains("http", na=False)]
        s_after  = s_after[~s_after.str.contains("http", na=False)]

        delim_before = s_before.str.contains(NOISE_PATTERN).mean() * 100 if len(s_before) else 0
        delim_after  = s_after.str.contains(NOISE_PATTERN).mean() * 100 if len(s_after) else 0
        delim_improve = round(delim_before - delim_after, 2)

    # DEDUPLICATION

        keys = dedupe_status.get("keys")
        executed = dedupe_status.get("executed", False)

        if keys:
            valid_keys = [k for k in keys if k in df_after.columns]
        else:
            valid_keys = []

        def dup_stats(df, subset=None):
            mask = df.duplicated(subset=subset, keep="first")
            return {
        "count": int(mask.sum()),
        "pct": round(mask.mean() * 100, 3)
    }

# BEFORE
        before_stats = dup_stats(df_before, valid_keys if valid_keys else None)
        dup_before = before_stats["pct"]
        dup_before_count = before_stats["count"]

# AFTER
        if executed:
            after_stats = dup_stats(df_after, valid_keys if valid_keys else None)
            dup_after = after_stats["pct"]
            dup_after_count = after_stats["count"]
        else:
            after_stats = None
            dup_after = np.nan
            dup_after_count = np.nan
            st.write("Deduplication not executed (aborted or skipped)")

# IMPROVEMENT
        if executed:
            dup_improve = round(max(dup_before - dup_after, 0), 2)
        else:
            dup_improve = 0.0   # no credit if step didn't run

# QUALITY SCORE (exclude dup_after if not executed)
        if executed:
            quality_score = round(100 - null_after - delim_after - dup_after, 2)
        else:
             quality_score = round(100 - null_after - delim_after, 2)

# RAW SCORE
        raw_score = (
    null_improve * 0.4 +
    delim_improve * 0.4 +
    dup_improve * 0.2
)
        raw_score = min(raw_score, 10)   # max raw improvement
        rows_before, rows_after, columns_before, columns_after = self.rows_columns_count(csvname,dbname,table_name)
        final_score = round(raw_score * 10, 2)
        summary_df = pd.DataFrame({
        "Metric": [
            "Null Percentage",
            "Delimiter Noise %",
            "Duplicate Rows %",
            "Rows",
            "Columns",
            "Overall Cleaning improvement Score",
            "Final Data Quality"
        ],
        "Before (%)": [
            round(null_before,6),
            delim_before,
            dup_before + 0.0 ,
            rows_before,
            columns_before,
            ' ',
            ' '
        ],
        "After (%)": [
            null_after,
            delim_after,
            dup_after,
            rows_after,
            columns_after,
            ' ',
            ' '

        ],
        "Improved (%)": [
            null_improve,
            delim_improve,
            dup_improve,
            abs(rows_before-rows_after),
            abs(columns_before-columns_after),
            final_score,
            quality_score

        ]
    })  

        return summary_df,quality_score

    def prepare_preview_and_full(self, df, preview_rows=1000, max_text_len=500):
    # full data (for download only)
        full_df = df.copy()

    # preview data (safe for display)
        preview_df = df.head(preview_rows).copy()

    # sanitize preview only (Arrow safety)
        for col in preview_df.select_dtypes(include=["object","string"]).columns:
            preview_df[col] = (
            preview_df[col]
            .astype(str)
            .str.slice(0, max_text_len)
        )

        return preview_df, full_df

    def show_dashboard(self, cleaned_csv,plot_csv,csvname,logger):
        # dataset_dbs = self.get_dataset_databases()
        tables = self.get_tables_from_database("test")
        dbname = "test"
    # check substring presence instead of exact prefix
        base = "transformed_" + cleaned_csv.lower()
        matched = [t for t in tables if base in t.lower()]
        if not matched:
            st.error(f"No MySQL table contains `{base}` in its name")
            return
        # dbname = matched[0]["db"]
        table_name = base
        df = self.load_table(dbname, table_name)
        if df is None or df.empty:
            st.warning(f"Table `{dbname}.{table_name}` returned no data.")
            return
         
        st.header(f"Data Quality Dashboard - {plot_csv.title()}")
        # KPI row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(df))
        c2.metric("Total Columns", len(df.columns))
        if any(c.lower()and c.lower().endswith("_end") and df[c].isna().any() for c in df.columns):
            st.markdown("**Structural NA in `*_end` is expected.**")
        c3.metric("Total Missing Cells", int(df.isna().sum().sum()))        
        st.markdown(f"## Data Validation Report ")        
        df_before = self.read_csv_with_fallback(DATA_DIR/csvname)
        cleaner = Datacleaner(df_before,csvname)
        data, dedupe_status = cleaner.remove_duplicates(csvname,logger)
        summary_df,quality_score = self.cleaning_score(df_before,df,dedupe_status,csvname,dbname,table_name)
        
        # safe_df = self.make_arrow_safe(summary_df)
        st.dataframe(summary_df, use_container_width=True)

        score = summary_df.loc[
        summary_df["Metric"] == "Overall Cleaning improvement Score",
            "Improved (%)"
        ].values[0]
        score = summary_df.loc[
        summary_df["Metric"] == "Final Data Quality",
            "Improved (%)"
        ].values[0]
        rows = summary_df.loc[
        summary_df["Metric"] == "Rows",
            "Improved (%)"
        ].values[0]
        columns = summary_df.loc[
        summary_df["Metric"] == "Columns",
            "Improved (%)"
        ].values[0]
        

        #missing values plot       
        assets = get_release_assets()
        plot_csv = plot_csv.replace(" ","_")
        before_url = self.find_images(assets,[plot_csv, "missing_values_output", "before"])
        after_url = self.find_images(assets,[plot_csv, "missing_values_output", "after"])

        if before_url and after_url:
            st.markdown("## Missing Values (Before vs After)")
            col1, col2 = st.columns(2)
            col1.image(before_url, width="stretch")
            col2.image(after_url, width="stretch")
        else:
            st.success("✔ No missing-values images available")

        before_out = self.find_images(assets,[plot_csv, "outliers", "before"])

        after_out = self.find_images(assets,[plot_csv, "outliers", "after"])

        if before_out and after_out:
            st.markdown("## Outliers (Before vs After)")
            col1, col2 = st.columns(2)
            col1.image(before_out[0], width="stretch")
            col2.image(after_out[0], width="stretch")
        else:
            st.success("✔ No outliers images available")

#         #bins if present 
        bins = self.find_images(assets,[plot_csv, "cleaned_bins"])

        if bins:
            st.markdown("## Bins Distribution")
            if len(bins) == 1:
                st.image(bins[0], width="stretch")
            else:
                idx = st.slider("Slide", 0, len(bins)-1, 0)
                st.image(bins[idx], width="stretch")
        else:
            st.info("✔ No bins images found")

        st.markdown(f"## Data")
        st.write(
    "ℹ️ This cleaned dataset is generated as part of an ETL demonstration. "
    "The original data was sourced from publicly available datasets and is "
    "provided here for educational and demonstration purposes only."
)
        preview_df, full_df = self.prepare_preview_and_full(df)
        
        c1, c2, c3 = st.columns(3)
        c1.download_button(
    "⬇ Download CSV (Full Data)",
    full_df.to_csv(index=False),
    f"{plot_csv}.csv",
    "text/csv"
)

# Excel (FULL DATA)
        buf = BytesIO()
        try:
            full_df.to_excel(buf, index=False)
            excel_bytes = buf.getvalue()
        except Exception:
            excel_bytes = None

        if excel_bytes:
            c2.download_button(
        "📊 Download Excel (Full Data)",
        excel_bytes,
        f"{plot_csv}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        else:
            c2.warning("Excel export not available")

# Text (FULL DATA but streamed safely)
        c3.download_button(
    "📄 Download Text (Full Data)",
    full_df.to_csv(sep="\t", index=False),
    f"{plot_csv}.txt",
    "text/plain"
)
        st.dataframe(preview_df, use_container_width=True)
        st.markdown("### 📊 Key Data Insights")
        st.write(df.describe())
        insights = [
    f"Final data quality score: {quality_score}",
    f"Total rows after cleaning: {len(df)}",
    f"Total columns: {len(df.columns)}"
]
        st.markdown("## 📄 Dashboard Report")

        pdf_url = self.find_dashboard_pdf(assets, cleaned_csv)
        if pdf_url:
            st.success("✅ Dashboard report available")

            st.markdown(f"🔗 **[Click here to download Dashboard Report (PDF)]({pdf_url})**")

        else:
            st.info("ℹ Dashboard report not generated yet")
    #     self.download_dashboard(
    #     csvname=csvname,
    #     summary_df=summary_df,
    #     quality_score=quality_score,
    #         insights=insights,
    #     output_dir=DASHBOARD_DIR
    # )

    def generate_pdf(self,csvname, summary_df, quality_score, insights, output_dir=DASHBOARD_DIR):
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_name = f"{csvname}_dashboard_{timestamp}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

    # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 40, "ETL Data Quality Dashboard Report")

    # Metadata
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 70, f"Dataset Name : {csvname}")
        c.drawString(40, height - 85, f"Generated : {timestamp}")
        c.drawString(40, height - 100, f"Data Quality Score : {quality_score}/100")

    # Key Insights
        y = height - 140
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Key Insights")
        y -= 20

        c.setFont("Helvetica", 10)
        for ins in insights:
            c.drawString(50, y, f"- {ins}")
            y -= 15

        y -= 20

    # Table
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Data Validation Report")
        y -= 22

# Table headers
        c.setFont("Helvetica-Bold", 10)
        headers = ["Metric", "Before", "After", "Improved"]

# Wider spacing between columns
        x= [40, 260, 350, 450]
        for h, px in zip(headers, x):
            c.drawString(px, y, h)

        y -= 15
        c.setFont("Helvetica", 9)

        for _, row in summary_df.iterrows():
            vals = [
            row["Metric"],
            row["Before (%)"],
            row["After (%)"],
            row["Improved (%)"]
        ]
            for v, px in zip(vals, x):
                c.drawString(px, y, str(v))
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 60
        c.save()
        return pdf_path


    def download_dashboard(self,csvname, summary_df, quality_score, insights, output_dir=DASHBOARD_DIR):
        
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_name = f"{csvname}_dashboard_{timestamp}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

    # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 40, "ETL Data Quality Dashboard Report")

    # Metadata
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 70, f"Dataset Name : {csvname}")
        c.drawString(40, height - 85, f"Generated : {timestamp}")
        c.drawString(40, height - 100, f"Data Quality Score : {quality_score}/100")

    # Key Insights
        y = height - 140
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Key Insights")
        y -= 20

        c.setFont("Helvetica", 10)
        for ins in insights:
            c.drawString(50, y, f"- {ins}")
            y -= 15

        y -= 20

    # Table
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Data Validation Report")
        y -= 22

# Table headers
        c.setFont("Helvetica-Bold", 10)
        headers = ["Metric", "Before", "After", "Improved"]

# Wider spacing between columns
        x= [40, 260, 350, 450]
        for h, px in zip(headers, x):
            c.drawString(px, y, h)

        y -= 15
        c.setFont("Helvetica", 9)

        for _, row in summary_df.iterrows():
            vals = [
            row["Metric"],
            row["Before (%)"],
            row["After (%)"],
            row["Improved (%)"]
        ]
            for v, px in zip(vals, x):
                c.drawString(px, y, str(v))
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 60
        c.save()

    # Streamlit download
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download Dashboard PDF",
                data=f,
            file_name=pdf_name,
            mime="application/pdf"
        )
        return pdf_path
    def open_dashboard(self, cleaned_csv, plot_csv, csvname, logger):
        st.session_state.page = "dashboard"
        st.session_state.cleaned_csv = cleaned_csv
        st.session_state.plot_csv = plot_csv
        st.session_state.csvname = csvname
        st.session_state.logger = logger
        # st.write("CSV stored in session:", st.session_state.csvname)
    def home_page(self,logger):
        st.title("📊 Automated Multi CSV - Dataset ETL & Analytics Dashboard")

    # Short description
        st.write(
        "An end-to-end pipeline that ingests multiple CSV files, "
        "performs data cleaning and transformation, loads the results "
        "into a cloud SQL database, and provides interactive dashboards "
        "and data quality reports per dataset." \
        "ℹ️ This cleaned dataset is generated as part of an ETL demonstration. "
    "The original data was sourced from publicly available datasets and is "
    "provided here for educational and demonstration purposes only." 
    )

        st.markdown("---")

    # Dataset section
        st.subheader("📁 Available Demo Datasets")
        st.write("Select a Demo dataset below to explore its dashboard and quality reports.")
        for f in DATA_DIR.glob("*.csv"):
            cleaned_stem = f.stem.lower().replace(" ", "_") + "_cleaned"  # for SQL
            plot_stem = f.stem.replace(" ", "_")  # for plots
            logger = logger
            st.button(f.stem, on_click=self.open_dashboard, args=(cleaned_stem, plot_stem,f.name,logger))
            # st.write("Found CSV file in folder:", f.name)

if __name__ == "__main__":
    import streamlit as st
    from pandasdatacleaning import Datacleaner
    from logger_config import get_logger 
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "cleaned_csv" not in st.session_state:
        st.session_state.cleaned_csv = None
    if "plot_csv" not in st.session_state:
        st.session_state.plot_csv = None
    if "csvname" not in st.session_state:
        st.session_state.csvname = None
    if "logger" not in st.session_state:
        st.session_state.logger = None
    dashboard = Datadashboard()
    if st.session_state.page == "home":
            dashboard.home_page(get_logger)

    elif st.session_state.page == "dashboard":
        # st.write("CSV passed to dashboard:", st.session_state.csvname)
        if st.sidebar.button("← Back to Home"):
            st.session_state.page = "home"
        else:
            dashboard.show_dashboard(st.session_state.cleaned_csv, st.session_state.plot_csv, st.session_state.csvname,st.session_state.logger)


    
