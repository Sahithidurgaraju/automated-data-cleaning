import streamlit as st
import pandas as pd,numpy as np
from sqlalchemy import create_engine
import os,json,glob,pytest,re
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit_carousel import carousel
from io import BytesIO
from logger_config import get_logger 
from pandasdatacleaning import Datacleaner
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data_output"
DATA_DIR = PROJECT_ROOT/"data"
PLOTS_DIR = PROJECT_ROOT / "plots"
DASHBOARD_DIR =  PROJECT_ROOT/"dashboard_reports"
@st.cache_data(show_spinner=False)
def sql_query(dbname, table_name):
    credential_file = os.path.join(os.path.dirname(__file__), "..", "sql_credentials", "sql_credentials.json")
    with open(credential_file) as f:
        database = json.load(f)

    mysqluri = f"mysql+pymysql://{database['user']}:{database['password']}@{database['localhost']}:{database['port']}"
    if not database["user"] or not database["password"] or not database["localhost"]:
        st.error("MySQL credentials are missing. Please update `sql_credentials.json` in the project root.")
        engine = None
    else:
        engine = create_engine(mysqluri)
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

        self.mysqluri = f"mysql+pymysql://{database['user']}:{database['password']}@{database['localhost']}:{database['port']}"
        if not database["user"] or not database["password"] or not database["localhost"]:
            st.error("MySQL credentials are missing. Please update `sql_credentials.json` in the project root.")
            self.engine = None
        else:
            try:
                self.engine = create_engine(self.mysqluri)
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
    
    def get_tables_from_databases(self,dataset_dbs):
        # Collect all tables from dataset DBs
        tables = []
        try:
            for db in dataset_dbs:
                tbls = pd.read_sql(f"SHOW TABLES FROM `{db}`", self.engine)
                for t in tbls.iloc[:, 0]:
                    tables.append({"db": db, "table": t})
            return tables
        except Exception as e:

            raise e

    def load_table(self, dbname, table_name):
        return sql_query(dbname, table_name)
    
    def rows_columns_count(self, csvname,dbname,table_name):
        df_before = pd.read_csv(DATA_DIR/csvname)
        df = self.load_table(dbname, table_name)
        df_after = pd.DataFrame(df)
        return rows_columns_count(df_before,df_after)

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

        NOISE_PATTERN = r"[;:/~–—]"

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
            '-',
            '-'
        ],
        "After (%)": [
            null_after,
            delim_after,
            dup_after,
            rows_after,
            columns_after,
            '-',
            '-'

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

    
    def show_dashboard(self, cleaned_csv,plot_csv,csvname,logger):
        dataset_dbs = self.get_dataset_databases()
        tables = self.get_tables_from_databases(dataset_dbs)
    # check substring presence instead of exact prefix
        base = cleaned_csv.lower()
        matched = [t for t in tables if base in t["table"].lower()]
        if not matched:
            st.error(f"No MySQL table contains `{cleaned_csv}` in its name")
            return
        dbname = matched[0]["db"]
        table_name = matched[0]["table"]
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
        df_before = pd.read_csv(DATA_DIR/csvname)
        cleaner = Datacleaner(df_before,csvname)
        data, dedupe_status = cleaner.remove_duplicates(csvname,logger)
        summary_df,quality_score = self.cleaning_score(df_before,df,dedupe_status,csvname,dbname,table_name)
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
        base_dir = os.path.abspath("plot")        
        before = list(PLOTS_DIR.glob(f"{plot_csv}_missing_values_output/{plot_csv}_missing_values_before_*.png"))
        after  = list(PLOTS_DIR.glob(f"{plot_csv}_missing_values_output/{plot_csv}_missing_values_after_*.png"))
    # 1. Missing values grouped slide ONLY if both exist
        slides= []
        if before and after:
           slides.append(("Missing Values (Before vs After)", before[0].resolve(), after[0].resolve()))
        if slides:
            if len(slides) == 1:
                title, *img_paths = slides[0]
                st.markdown(f"## {title}")
                if len(img_paths) == 2:  # missing before/after
                    col1, col2 = st.columns(2)
                    img1 = Image.open(img_paths[0]).resize((900, 600))
                    img2 = Image.open(img_paths[1]).resize((900, 600))
                    col1.image(img1)
                    col2.image(img2)
                else:  # single image slide
                    img = Image.open(img_paths[0]).resize((900, 600))
                    st.image(img)
                    st.markdown("**ℹ Structural NA is expected if shown in *_end column**")
            else:
                idx = st.slider("Slide", 0, len(slides)-1, 0)
                title, img_path = slides[idx]
                st.markdown(f"## {title}")
                img = Image.open(img_path).resize((900, 600))
                st.image(img)
        else:
            st.success("✔ No image slides available")
        #outliers
        outliers = []
        before_outliers = list(PLOTS_DIR.glob(f"{plot_csv}_outliers/{plot_csv}_outliers_before_*.png"))
        after_outliers = list(PLOTS_DIR.glob(f"{plot_csv}_outliers/{plot_csv}_outliers_after_*.png"))        
        if before_outliers and after_outliers:
            outliers.append(("Outliers (Before vs After)", before_outliers[0].resolve(), after_outliers[0].resolve()))
        if outliers:
            if len(outliers) == 1:
                title, *img_paths = outliers[0]
                st.markdown(f"## {title}")
                if len(img_paths) == 2:  # missing before/after
                    col1, col2 = st.columns(2)
                    img1 = Image.open(img_paths[0]).resize((900, 600))
                    img2 = Image.open(img_paths[1]).resize((900, 600))
                    col1.image(img1)
                    col2.image(img2)
                else:  # single image slide
                    img = Image.open(img_paths[0]).resize((900, 600))
                    st.image(img)
                    st.markdown("**ℹ Structural NA is expected if shown in year*_end column**")
            else:
                idx = st.slider("Slide", 0, len(outliers)-1, 0)
                title, img_path = outliers[idx]
                st.markdown(f"## {title}")
                img = Image.open(img_path).resize((900, 600))
                st.image(img)
        else:
            st.success("✔ No image slides available")
        #bins if present 
        bins = []
        bn = list(PLOTS_DIR.glob(f"{plot_csv.lower()}_cleaned_*/{plot_csv.lower()}_cleaned_*.png"))
        if bn:
            for i in bn: bins.append(("Bins Distribution", i.resolve()))
            if bins:
    # If only 1 slide → show normally (no slider error)
                if len(bins) == 1:
                    title, img_path = bins[0]  # 
                    st.markdown(f"## {title}")
                    st.image(Image.open(img_path).resize((900, 600)))
                else:
                    idx = st.slider("Slide", 0, len(bins)-1, 0)
                    title, img_path = bins[idx]  # 
                    st.markdown(f"## {title}")
                    st.image(Image.open(img_path).resize((900, 600)))
            else:
                st.info("✔ No bins images found")
        #bins if present 
        st.markdown(f"## Data")
        

        buf = BytesIO()
        df.to_excel(buf, index=False)
        excel_bytes = buf.getvalue()

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ CSV", df.to_csv(index=False).encode(), f"{plot_csv}.csv", "text/csv")
        c2.download_button("📊 Excel", excel_bytes, f"{plot_csv}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c3.download_button("📄 Text", df.to_string(), f"{plot_csv}.txt", "text/plain")

        st.dataframe(df)
        st.markdown("### 📊 Key Data Insights")
        st.write(df.describe(include="all"))
        insights = [
    f"Final data quality score: {quality_score}",
    f"Total rows after cleaning: {len(df)}",
    f"Total columns: {len(df.columns)}"
]
        st.markdown(f"## Dashboard Report")
        self.download_dashboard(
        csvname=csvname,
        summary_df=summary_df,
        quality_score=quality_score,
            insights=insights,
        output_dir=DASHBOARD_DIR
    )

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
        st.title("📁 Available Datasets")
        for f in DATA_DIR.glob("*.csv"):
            cleaned_stem = f.stem.lower().replace(" ", "_") + "_cleaned"  # for SQL
            plot_stem = f.stem  # for plots
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


    