import subprocess
from pathlib import Path
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
CLEANED_DATA_DIR = Path("cleaned_data_output")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Filter CSV files
csv_files = [f for f in os.listdir(CLEANED_DATA_DIR) if f.lower().endswith(".csv")]

for csv_file in csv_files:
    report_name = REPORTS_DIR / f"{Path(csv_file).stem}_report.html"
    print(f"Generating report for {csv_file} → {report_name.name}")
    
    # Run pytest, forcing the fixture to only pick this CSV via environment variable
    env = os.environ.copy()
    env["PYTEST_CURRENT_CSV"] = str(CLEANED_DATA_DIR / csv_file)
    
    cmd = f'python -m pytest tests/test_apply_transformation.py --html="{report_name}" --self-contained-html'
    
    subprocess.run(cmd, shell=True, env=env)