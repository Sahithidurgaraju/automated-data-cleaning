## Automated Multi CSV - Dataset ETL & Analytics Dashboard
## (Data Cleaning --> Cleaning Validation --> Transformation --> MYSQL)

## Project Summary:

This project implements a fully automated, analytics-focused ETL pipeline built in Python to **clean, validate, transform, and load large multi-CSV datasets** into a **MySQL-compatible distributed database (TiDB Cloud)**.

The pipeline is **optimized for local execution on laptops**, leveraging **vectorized Pandas operations, chunk-based processing, and batched SQL inserts** to deliver **high performance, memory safety, and execution stability**, even when processing large, wide datasets.

**All stages of the pipeline — data cleaning, validation, transformation, and database ingestion — run end-to-end without manual intervention**, automatically generating **structured artifacts, validation reports, and analytics-ready outputs** suitable for downstream analysis and dashboarding.


## 🚀 Key Features

• **Supports multiple CSV files in a single run**, enabling batch processing of datasets  

• **Automatically creates one database per dataset** (derived from the CSV filename) if it does not already exist  

• **Normalizes multi-value cells (`;` → `,`) without exploding rows**, preserving row counts and dataset integrity  

• **Uses vectorized Pandas column operations instead of slow row-wise loops** for high-performance data cleaning  

• **Batched SQL inserts** to ensure **safe, memory-efficient local execution** on laptops  

• **Internal database-level validation using set-based SQL queries**, avoiding expensive row scans  

• **Exports machine-readable validation reports (JSON)** with clear **pass/fail status and detailed failure reasons**  

• **Designed to process ~100K rows × 60 columns in ~2 minutes** across multiple CSV datasets  

• **Fully automated artifact lifecycle management**, where **old outputs are replaced per dataset** to prevent disk bloat and stale results  


## Repository Structure

**data/**                      → Input CSV files

**cleaned_data_output/**       → Cleaned CSV outputs (same filename preserved)

**validation_reports/**        → Validation reports (JSON)

**plots/**                     → missing plots, outliers, bins histogram

**logs/**                     → dataset wise logs 

**json_output/**               → Auto-generated schema & transformation metadata

**sql_credentials/**            → Database connection configuration  

**src/etldashboard.py**         → Streamlit dashboard  

**run_reports.py**             → Cleaning + validation execution  

**generate_transformation.py** → Auto-generate transformation configs 

**apply_transformation.py**    → Apply transforms & push to database

**Jenkinsfile**                → CI pipeline configuration  

## Project Architecture:


<img width="975" height="650" alt="image" src="https://github.com/user-attachments/assets/1f03977d-2558-4122-8622-8791d3e87793" />





## ⚙️ Challenges & Optimization Journey

• Initially used **`pandas.apply()` for text cleaning**, but found it **took hours** when applied across **60 text-heavy columns**

• Explored **parallel execution for text transformations**, but observed **limited gains** due to Python overhead and I/O constraints  

• Tested **Swifter to optimize `apply` performance**, but results were **still not suitable for large-scale datasets**  

• Introduced **chunk-based processing for CSV read/write operations** to **prevent RAM spikes and system instability**  

• After experimentation and research, confirmed that:

  • **Vectorized Pandas column operations combined with set-based MySQL validation** provide the **best balance of speed, memory efficiency, and execution stability**, enabling **laptop-safe processing without row explosion**


## Technologies & Skills Used:

•	**Python**: Pandas, NumPy, SQLAlchemy, PyMySQL, Pytest

•	**Database**: TiDB Cloud (MySQL-compatible)

•	**Visualization**: Matplotlib

•	**Automation**: Jenkins CI

•	**Config & Reporting**: JSON-based validation and metadata

•	**Optimization**: Vectorized operations, chunking, batched inserts

## 📊 Binning Visualization Support

• **Automatically detects numeric columns** and generates `*_bin` columns when binning is enabled  

• **Generates one plot per CSV dataset**, with **auto-updated output and scalable figure sizing**  

• **Uses only vectorized count-based summaries for plotting**, avoiding row-wise operations and ensuring efficient visualization even for large datasets  

## 📂 Dataset Notes

• This project uses **publicly available and demo datasets** representing a variety of real-world data quality patterns (missing values, mixed formats, outliers, and high-cardinality fields).

• Datasets are used **solely for demonstration and educational purposes** to showcase pipeline design, performance optimization, and validation logic.

## How a user will run this project:

## Installation

•	Install **Python**

•	Install required Python packages including:

•	**Pandas** → data cleaning & transformation

•	**NumPy** → memory-efficient numeric operations

•	**SQLAlchemy + PyMySQL** → MySQL ingestion

•	**Cryptography** → required for secure MySQL authentication (caching_sha2_password)

•	**Pytest** → automated testing & report generation

## 🔐 GitHub Token Requirement

This project integrates with the **GitHub Releases API** to automatically manage JSON artifacts generated during the ETL process, including:

- Schema validation outputs  
- Transformation configuration files  
- Dataset-level validation reports  

To enable this functionality, a **GitHub Personal Access Token (PAT)** is required.



### Why the GitHub Token Is Required

The token is used to:
- Create or access GitHub releases
- Delete outdated dataset-specific JSON artifacts
- Upload newly generated JSON files as versioned release assets
- Avoid unauthenticated API rate limits during automation



### 📌 When a GitHub Token Is Required

| Execution Mode | GitHub Token Required |
|---------------|----------------------|
| Full ETL pipeline execution (local or CI) | ✅ Yes |
| Jenkins / automated pipeline runs | ✅ Yes |
| Test suite execution | ✅ Yes |
| Streamlit dashboard (read-only analytics) | ❌ No |

> The Streamlit dashboard does **not** perform GitHub uploads and will run without a GitHub token.



###  How to Set Up the GitHub Token

#### Step 1: Create a GitHub Personal Access Token

1. Go to **GitHub → Settings → Developer settings → Personal access tokens**
2. Generate a token with the following permissions:
   - **Contents: Read & Write**
   - **Releases: Read & Write**
3. Copy the token and store it securely



#### Step 2: Set the Token as an Environment Variable

**Linux / macOS**

    ```bash
    export GITHUB_TOKEN=your_token_here
    
**Windows Powershell**

   setx GITHUB_TOKEN "your_token_here"
  
## Clone the repository

      git clone https://github.com/Sahithidurgaraju/automated-data-cleaning

## Run this command to install everything:

      pip install -r requirements.txt

## Run the project in this sequence

1. Place one or more CSV files inside the `data/` folder.

2. Run the reports script to clean the data - **python run_reports.py**:
   
   - Each file is processed column-wise using vectorized Pandas operations.
   - Cleaned output is stored in `cleaned_data_output/` using the original CSV filename (no new rows are created or removed).
   - A structural validation report is generated and exported as JSON.

3. Generate the transformation configuration - **python generate_transformation.py**:

   - The script reads the cleaned CSV files from `cleaned_data_output/`.
   - Produces a user-editable `transform_config.json` file for each dataset.

4. Apply transformations and push to MySQL - **python apply_transformation.py**:

   - Transformations (casting, binning, MLops if enabled) are applied to the cleaned data.
   - Final transformed datasets are inserted into MySQL using batched SQL writes for laptop-safe execution.


## After that run these commands:

python run_reports.py                # Step 1: Clean data & generate validation JSON 

python generate_transformation.py    # Step 2: Generate user-editable transform config JSON

python apply_transformation.py       # Step 3: Apply transformations (bins, MLops if enabled) and push to MySQL

python dashboard_pdf.py              # Step 4: Generate data quality report

python cleanup_upload_images.py      # Step 5: cleanup old images and upload new images in github releases artifacts

> ⚠️ Step 5 requires a valid `GITHUB_TOKEN` to upload artifacts to GitHub Releases.

## MySQL Connection Note

-To connect **MySQL** successfully:

-The pipeline uses **PyMySQL** driver, which requires the **cryptography** package for secure authentication plugins.

-If **cryptography** is missing, **MySQL** connection fails. 

-Installing it ensures secure connectivity and avoids auth runtime errors.

## Database Backend

- This project uses **TiDB Cloud**, a MySQL-compatible distributed database.

    - Connection via SQLAlchemy + PyMySQL

    - Standard MySQL SQL syntax

- **Works unchanged with**:

    - MySQL

    - TiDB Cloud

    - Amazon RDS MySQL

    - Azure MySQL

- **TiDB Cloud provides**:

    - Serverless operation

    - Horizontal scalability

    - Production-like database behavior
  
## Automatic Artifact Management:

•	The pipeline processes input **CSV files** placed in the data/ folder.

•	Cleaned output is stored in **cleaned_data_output/** using the original CSV filename.

•	When the same CSV dataset is re-run, the system automatically:

     •	Removes old bin plots, audit JSON reports, and logs for that dataset.

     •	Generates and stores new artifacts.

•	Keeps only the latest outputs, preventing disk bloat and repeated memory allocation.

•	Artifacts from other CSV datasets are preserved to maintain lineage.

•	This ensures **fast, stable, and memory-safe continuous local execution, ideal** for dashboard preparation.

## Data CLeaning Process:

## Before Cleaning:

## Missing values Plot of messy data:

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/8035bc2c-5b42-44c9-b47f-735e7fcb2fdf" />

## Outliers plot of messy data:

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/dceba6e8-2dc4-41be-91e7-210443089e4d" />


## After Cleaning:

## Missing values Plot of messy data:

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/b3f303f2-5374-43c3-aa3c-4613fd35af21" />

## Outliers plot of messy data:

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/9489bbcb-2e66-4e5f-9b2e-e55604f55989" />

## Validate report Sample Artifacts:

{
  "dataset": "messydata",
  "status": "PASS",
  "checks": {
    "non_empty": {
      "status": "PASS",
      "message": "Dataset contains 20 rows after cleaning"
    },
    "schema_match": {
      "status": "PASS",
      "message": "All expected schema columns are present"
    },
    "deduplication": {
      "status": "PASS",
      "message": "Deduplication skipped intentionally: No valid deduplication keys"
    },
    "no_nulls": {
      "status": "PASS",
      "message": "Only structural nulls found (6); expected"
    }
  },
  "timestamp": "20251224_1224"
}

## 🧩 Only Structural NULLs Found

When a **year column contains mixed formats** such as `2023–2024` and single values like `2023`, the pipeline **splits year ranges into `year` and `year_end` columns**.

For **single-year values**, the `year_end` field **remains empty or contains whitespace**, which is **intentionally treated as a structural NULL**, as **no valid end range exists**.

These **NULL values are expected, acceptable, and explicitly excluded from failure assertions**, since they **represent a valid business condition rather than bad or missing data**.


## Transformation Stage:

## Transform Config JSON (User-Editable):

{
    "dataset": "messydata_cleaned",
    "columns": {
        "rank": {
            "enabled": {
                "cast_float": true,
                "bins": true,
                "scale": false,
                "log": false,
                "outliers": true
            }
        },
        "peak": {
            "enabled": {
                "cast_float": true,
                "bins": false,
                "scale": false,
                "log": true,
                "outliers": false
            }
        },
        "all_time_peak": {
            "enabled": {
                "cast_float": true,
                "bins": false,
                "scale": false,
                "log": true,
                "outliers": false
            }
        },
        "actual_gross": {
            "enabled": {
                "cast_float": true,
                "bins": true,
                "scale": true,
                "log": true,
                "outliers": true
            }
        },
        "adjusted_gross_in_2022_dollars": {
            "enabled": {
                "cast_float": true,
                "bins": true,
                "scale": true,
                "log": true,
                "outliers": true
            }
        },
        "artist": {
            "enabled": {
                "lowercase": true,
                "strip": true,
                "normalize_delimiters": false,
                "split_year": false
            }
        },
        "tour_title": {
            "enabled": {
                "lowercase": true,
                "strip": true,
                "normalize_delimiters": true,
                "split_year": false
            }
        },
        "year_s": {
            "enabled": {
                "lowercase": true,
                "strip": true,
                "normalize_delimiters": false,
                "split_year": true
            }
        },
        "shows": {
            "enabled": {
                "cast_float": true,
                "bins": true,
                "scale": false,
                "log": true,
                "outliers": true
            }
        },
        "average_gross": {
            "enabled": {
                "cast_float": true,
                "bins": true,
                "scale": true,
                "log": true,
                "outliers": true
            }
        },
        "year_s_start": {
            "enabled": {
                "cast_int": true,
                "bins": true,
                "scale": true,
                "log": false,
                "outliers": true
            }
        },
        "year_s_end": {
            "enabled": {
                "cast_int": true,
                "bins": true,
                "scale": true,
                "log": false,
                "outliers": true
            }
        }
    }
}

## Bin Distribution Plots:

<img width="1200" height="600" alt="messydata_cleaned_20251224_122450_bins_20251225_151749" src="https://github.com/user-attachments/assets/27c383ad-f58c-4184-9751-735ec56eaf0a" />

## MySQL Push:

<img width="1918" height="577" alt="image" src="https://github.com/user-attachments/assets/39d070d9-4154-437b-a4c4-3266cd7f5b55" />

## CI Automation (Jenkins)

**The repository includes a Jenkinsfile to enable**:

- Automated dependency installation

- Full ETL pipeline execution

- Test execution

- Artifact archival (reports, plots, logs)

- This allows scheduled or trigger-based ETL runs in CI environments.
  
## Dashboard Visualization

**After pipeline execution, launch the dashboard**:

       streamlit run src/etldashboard.py

**The dashboard provides**:

- Dataset-wise validation status

- Missing value summaries

- Bin distribution insights

- Overall dataset health metrics

**The dashboard is read-only and does not trigger ETL execution.**

## 📈 BI Validation (Like Power BI & Tableau)

After completing the full ETL pipeline and loading the cleaned, validated data into **MySQL (TiDB Cloud)**, the final datasets were tested in **Power BI and Tableau** to verify **analytics readiness and downstream usability**.

### What Was Validated

• **Schema consistency and data types**, ensuring fields were correctly interpreted by BI tools without manual fixes  

• **Row count integrity**, confirming no row loss or unintended duplication during cleaning and transformation  

• **NULL handling and structural NULL logic**, validating that expected structural NULLs did not break visualizations or aggregations  

• **Numeric aggregation correctness**, ensuring measures behaved correctly in filters, bins, and groupings  

• **End-to-end compatibility**, demonstrating that the pipeline output is immediately usable for dashboarding and reporting  

### Outcome

• The datasets loaded seamlessly into **Power BI and Tableau** without additional preprocessing  

• Visualizations rendered correctly, confirming **clean schema design and transformation integrity**  

• This validation step confirms the pipeline produces **analytics-ready data**, suitable for real-world BI and decision-making workflows  

> The screenshots below demonstrate successful dashboard creation and data exploration using the pipeline outputs.

### Power BI Validation

![powerbi_validation_with_cleaned_data](https://github.com/user-attachments/assets/f6b4b45f-5cff-47fd-ab2d-ecde8f414ad3)


### Tableau Validation

![Tableau_validation_with_cleaned_data](https://github.com/user-attachments/assets/2887bf6c-418a-4d3c-a3f2-064aa4b28b0b)


## Future Enhancements

-Support for **Excel file automation (.xlsx, .xls)** alongside CSV ingestion:

   - Multi-sheet handling

   - Schema inference per sheet

   - Large Excel file chunked processing

-**Cloud-native deployment support**

   - Containerization using Docker

   - Execution on cloud compute (AWS EC2 / Azure VM / GCP Compute Engine)

   - Managed database integrations (RDS, Cloud SQL)

-**Scheduled and event-driven ETL execution**

   - Cron-based scheduling

   - CI-triggered runs (Jenkins)

-**BI and analytics integrations**

   - Direct connectors to Metabase, Power BI, Tableau

- Schema drift detection and automated alerts

## Author

**Sahithi** — Aspiring Data Engineer

Developer of the ETL Automation Pipeline

## Usage Notice

This project is for demonstration and personal use. 

All rights are reserved by the author.

