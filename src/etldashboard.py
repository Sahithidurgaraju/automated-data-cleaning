import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os,json,glob,pytest,re
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_carousel import carousel
from io import BytesIO
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data_output"
DATA_DIR = PROJECT_ROOT/"data"
PLOTS_DIR = PROJECT_ROOT / "plots"
# from src.config import DATA_DIR,JSON_DIR,CLEANED_DATA_DIR,PLOTS_DIR
class Datadashboard:
    def __init__(self):
        credential_file = os.path.join(os.path.dirname(__file__), "..", "sql_credentials", "sql_credentials.json")
        with open(credential_file) as f:
            database = json.load(f)

        self.mysqluri = f"mysql+pymysql://{database['user']}:{database['password']}@{database['localhost']}:{database['port']}"
        self.engine = create_engine(self.mysqluri)

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
        return pd.read_sql(f"SELECT * FROM `{dbname}`.`{table_name}`", self.engine)


    def show_dashboard(self, cleaned_csv,plot_csv):
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
        if any(c.lower().startswith("year") and c.lower().endswith("_end") and df[c].isna().any() for c in df.columns):
            st.markdown("**Structural NA in `*_end` is expected.**")
        c3.metric("Total Missing Cells", int(df.isna().sum().sum()))
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
                    st.markdown("**ℹ Structural NA is expected if shown in year*_end column**")
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

    # def go_dashboard(self,csvname):
    #     st.session_state.page = "dashboard"
    #     st.session_state.csvname = csvname
    def open_dashboard(self, cleaned_csv, plot_csv):
        st.session_state.page = "dashboard"
        st.session_state.cleaned_csv = cleaned_csv
        st.session_state.plot_csv = plot_csv
    def home_page(self):
        st.title("📁 Available Datasets")
        for f in DATA_DIR.glob("*.csv"):
            cleaned_stem = f.stem.lower().replace(" ", "_") + "_cleaned"  # for SQL
            plot_stem = f.stem  # for plots
            st.button(f.stem, on_click=self.open_dashboard, args=(cleaned_stem, plot_stem))

if __name__ == "__main__":
    import streamlit as st
    
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "cleaned_csv" not in st.session_state:
        st.session_state.cleaned_csv = None
    if "plot_csv" not in st.session_state:
        st.session_state.plot_csv = None
    dashboard = Datadashboard()
    if st.session_state.page == "home":
            dashboard.home_page()

    elif st.session_state.page == "dashboard":
        if st.sidebar.button("← Back to Home"):
            st.session_state.page = "home"
        else:
            dashboard.show_dashboard(st.session_state.cleaned_csv, st.session_state.plot_csv)