## CSV ETL AUTOMATION 
## (Data Cleaning --> Cleaning Validation --> Transformation --> MYSQL)

## Project Summary:

This project implements a fully automated, analytics-focused ETL pipeline built in Python to clean, validate, transform, and load large multi-CSV datasets into a MySQL-compatible distributed database (TiDB Cloud). The pipeline is optimized for local execution on laptops, using vectorized Pandas operations, chunk-based processing, and batched SQL inserts to ensure high performance, memory safety, and execution stability — even for large datasets. All stages (cleaning, validation, transformation, and database ingestion) run end-to-end without manual intervention, generating structured artifacts and validation reports automatically.

## Key Features:

•	Supports multiple CSV files in a single run

•    Automatically creates one database per dataset (based on CSV filename) if missing

•    Normalizes multi-value cells (; → ,) without exploding rows

•    Uses vectorized Pandas column operations instead of row-wise loops

•    Batched SQL inserts for safe local execution

•    Internal database-level validation using set-based SQL queries

•    Exports machine-readable validation reports (JSON) with pass/fail status and failure reasons

•    Designed to process 1L rows × 60 columns in ~2 minutes across multiple CSVs

•    Fully automated artifact lifecycle management (old outputs are replaced per dataset)

## Repository Structure

data/                      → Input CSV files

cleaned_data_output/       → Cleaned CSV outputs (same filename preserved)

validation_reports/        → Validation reports (JSON)

plots/                     → missing plots, outliers, bins histogram

logs/                      → dataset wise logs 

json_output/               → Auto-generated schema & transformation metadata

sql_credentials/            → Database connection configuration  

src/etldashboard.py         → Streamlit dashboard  

run_reports.py             → Cleaning + validation execution  

generate_transformation.py → Auto-generate transformation configs 

apply_transformation.py    → Apply transforms & push to database

run_pipeline.py            → Single entry-point for full automation 

Jenkinsfile                → CI pipeline configuration  

## Project Architecture:


<img width="975" height="650" alt="image" src="https://github.com/user-attachments/assets/1f03977d-2558-4122-8622-8791d3e87793" />





## Challenges & Optimization Journey

•	Initially used pandas.apply() for text cleaning but found it takes hours for 60 text-heavy columns 

•	Explored parallel execution for text transformations 

•	Tested Swifter to optimize apply speed, but still not ideal for huge scale 

•	Introduced chunk-based processing while reading/writing CSV to prevent RAM spikes 

•	After research, confirmed:

     •Vectorized Pandas column operations + set-based MySQL validation is the best approach. It delivers fast execution, memory optimization, and stability on laptops without row explosion.

## Technologies & Skills Used:

•	Python: Pandas, NumPy, SQLAlchemy, PyMySQL, Pytest

•	Database: TiDB Cloud (MySQL-compatible)

•	Visualization: Matplotlib

•	Automation: Jenkins CI

•	Config & Reporting: JSON-based validation and metadata

•	Optimization: Vectorized operations, chunking, batched inserts

## Binning Visualization Support:

•	Detects numeric columns and generates *_bin columns when enabled

•	Plot is generated once per CSV, auto-updated, scalable figsize

•	Uses only vectorized count summaries for plotting

## How a user will run this project:

## Installation

•	Install Python

•	Install required Python packages including:

•	Pandas → data cleaning & transformation

•	NumPy → memory-efficient numeric operations

•	SQLAlchemy + PyMySQL → MySQL ingestion

•	Cryptography → required for secure MySQL authentication (caching_sha2_password)

•	Pytest → automated testing & report generation

## Run this command to install everything:

pip install pandas numpy pytest streamlit SQLAlchemy PyMySQL pyarrow cryptography matplotlib seaborn pillow reportlab openpyxl pycountry rapidfuzz

## Run the project in this sequence

1. Place one or more CSV files inside the `data/` folder.

2. Run the reports script to clean the data:
   - Each file is processed column-wise using vectorized Pandas operations.
   - Cleaned output is stored in `cleaned_data_output/` using the original CSV filename (no new rows are created or removed).
   - A structural validation report is generated and exported as JSON.

3. Generate the transformation configuration:
   - The script reads the cleaned CSV files from `cleaned_data_output/`.
   - Produces a user-editable `transform_config.json` file for each dataset.

4. Apply transformations and push to MySQL:
   - Transformations (casting, binning, MLops if enabled) are applied to the cleaned data.
   - Final transformed datasets are inserted into MySQL using batched SQL writes for laptop-safe execution.


## After that run these commands:

python run_reports.py         # Step 1: Clean data & generate validation JSON 

python generate_transformation.py    # Step 2: Generate user-editable transform config JSON

python apply_transformation.py       # Step 3: Apply transformations (bins, groupby, filters if enabled) and push to MySQL

## MySQL Connection Note

-To connect MySQL successfully:

-The pipeline uses PyMySQL driver, which requires the cryptography package for secure authentication plugins.

-If cryptography is missing, MySQL connection fails. 

-Installing it ensures secure connectivity and avoids auth runtime errors.

## Database Backend

- This project uses TiDB Cloud, a MySQL-compatible distributed database.

    - Connection via SQLAlchemy + PyMySQL

    - Standard MySQL SQL syntax

- Works unchanged with:

    - MySQL

    - TiDB Cloud

    - Amazon RDS MySQL

    - Azure MySQL

- TiDB Cloud provides:

    - Serverless operation

    - Horizontal scalability

    - Production-like database behavior
  
## Automatic Artifact Management:

•	The pipeline processes input CSV files placed in the data/ folder.

•	Cleaned output is stored in cleaned_data_output/ using the original CSV filename.

•	When the same CSV dataset is re-run, the system automatically:

     •	Removes old bin plots, audit JSON reports, and logs for that dataset.

     •	Generates and stores new artifacts.

•	Keeps only the latest outputs, preventing disk bloat and repeated memory allocation.

•	Artifacts from other CSV datasets are preserved to maintain lineage.

•	This ensures fast, stable, and memory-safe continuous local execution, ideal for dashboard preparation.

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

## Only structural nulls found:

When a year column contains mixed formats such as 2023-2024 and single values like 2023, the pipeline splits ranges into year and year_end.

For single year values, the year_end part remains empty or contains spaces, which is intentionally treated as a structural NULL because no valid end range exists.

These NULLs are expected, acceptable, and excluded from failure assertions, as they do not represent bad data but a valid business condition.

## Transformation Stage:

## Transform Config JSON (User-Editable):

{
    "dataset": "messydata_cleaned_20251224_122450",
    "columns": {
        "rank": {
            "suggested": {
                "cast": {
                    "type": "float",
                    "reason": "Numeric column",
                    "default": true
                },
                "bins": {
                    "type": "auto",
                    "reason": "High cardinality numeric column",
                    "default": false
                }
            },
            "enabled": {
                "cast": true,
                "bins": true
            }
        }
    "dataset_ops": {
        "filter": {
            "rank": {
                ">=": null,
                "<=": null           
        },
        "groupby": {
            "by": [],
            "agg": {}
        }
    }
}

## Bin Distribution Plots:

<img width="1200" height="600" alt="messydata_cleaned_20251224_122450_bins_20251225_151749" src="https://github.com/user-attachments/assets/27c383ad-f58c-4184-9751-735ec56eaf0a" />

## MySQL Push:

<img width="1918" height="577" alt="image" src="https://github.com/user-attachments/assets/39d070d9-4154-437b-a4c4-3266cd7f5b55" />


## Clone the repository

git clone https://github.com/Sahithidurgaraju/automated-data-cleaning

## CI Automation (Jenkins)

The repository includes a Jenkinsfile to enable:

- Automated dependency installation

- Full ETL pipeline execution

- Test execution

- Artifact archival (reports, plots, logs)

- This allows scheduled or trigger-based ETL runs in CI environments.
  
## Dashboard Visualization

After pipeline execution, launch the dashboard:

       streamlit run src/etldashboard.py
The dashboard provides:

- Dataset-wise validation status

- Missing value summaries

- Bin distribution insights

- Overall dataset health metrics

The dashboard is read-only and does not trigger ETL execution.

## Future Enhancements

-Support for Excel file automation (.xlsx, .xls) alongside CSV ingestion:

   - Multi-sheet handling

   - Schema inference per sheet

   - Large Excel file chunked processing

- Cloud-native deployment support:

   - Containerization using Docker

   - Execution on cloud compute (AWS EC2 / Azure VM / GCP Compute Engine)

   - Managed database integrations (RDS, Cloud SQL)

- Scheduled and event-driven ETL execution

   - Cron-based scheduling

   - CI-triggered runs (Jenkins)

- BI and analytics integrations

   - Direct connectors to Metabase, Power BI, Tableau

- Schema drift detection and automated alerts

## Author

**Sahithi** — Developer of the ETL Automation Pipeline

## Usage Notice

This project is for demonstration and personal use. 

All rights are reserved by the author.

