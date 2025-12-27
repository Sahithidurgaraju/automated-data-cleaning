## CSV ETL AUTOMATION (DATA CLEANING --> CLEANING VALIDATION --> TRANSFORM --> MYSQL)

## Project Summary:

An automated analytics-focused ETL pipeline built in Python to clean, transform, and load large multi-CSV datasets into MySQL, followed by structural validation reports exported to JSON.
Designed to run efficiently on a laptop without crashing or increasing rows, using vectorized Pandas column operations and batched SQL inserts.

## Key Features:

•	Supports multiple CSV files and stores cleaned data by CSV filename

•	Automatically creates MySQL database per CSV name if missing

•	Normalizes multi-value cells (; → ,) without exploding rows

•	Uses vectorized Pandas string operations instead of slow row loops

•	Experimented with parallel text cleaning, Swifter, and chunk processing

•	Validated column-wise NULL %, row count match, and transform integrity inside MySQL

•	Exports validation status JSON (pass/fail) with specific failure reasons

•	Execution optimized to process 1L rows × 60 columns in ~2 minutes across multiple CSVs

## Repository Structure
data/                      → Input CSV files

cleaned_data_output/       → Cleaned CSV outputs (same filename preserved)

validation_reports/        → Validation reports (JSON)

plots/                     → missing plots, outliers, bins histogram

logs/                      → dataset wise logs 

json_output                → here detect the column types and apply column types, transformation json 

sql_credentials            → my sql workbench credentials

run_reports.py             → Executes cleaning + generates validation JSON

generate_transformation.py → Generates user-editable transformation config JSON

apply_transformation.py    → Applies transforms + pushes final data to MySQL

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

•	Python (Pandas, NumPy, SQLAlchemy, PyMySQL,Pytest)

•	MySQL (Set-based aggregation validation, internal scans)

•	Matplotlib (Binning visualization, scalable figure sizing)

•	JSON automation (Dynamic validation & transform config generation)

•	Optimized processing using vectorized column operations for speed and memory safety

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

pip install pandas numpy sqlalchemy pymysql cryptography pytest

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
   - Transformations (casting, binning, filtering, groupby if enabled) are applied to the cleaned data.
   - Final transformed datasets are inserted into MySQL using batched SQL writes for laptop-safe execution.


## After that run these commands:

python run_reports.py         # Step 1: Clean data & generate validation JSON 

python generate_transformation.py    # Step 2: Generate user-editable transform config JSON

python apply_transformation.py       # Step 3: Apply transformations (bins, groupby, filters if enabled) and push to MySQL

## MySQL Connection Note

To connect MySQL successfully:

The pipeline uses PyMySQL driver, which requires the cryptography package for secure authentication plugins.

If cryptography is missing, MySQL connection fails. 

Installing it ensures secure connectivity and avoids auth runtime errors.

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

## Future Enhancements

- Add scheduling support for automated daily ETL runs

- Cloud execution support (AWS Lambda, Airflow)

- MySQL → BI dashboard connectors (Metabase, Power BI, Tableau)

## Author

**Sahithi** — Developer of the ETL Automation Pipeline

## Usage Notice

This project is for demonstration and personal use. 

All rights are reserved by the author.

