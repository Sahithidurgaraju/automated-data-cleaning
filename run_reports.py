import subprocess
from pathlib import Path
import os

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Filter CSV files
csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]

for csv_file in csv_files:
    report_name = REPORTS_DIR / f"{Path(csv_file).stem}_report.html"
    print(f"Generating report for {csv_file} → {report_name.name}")
    
    # Run pytest, forcing the fixture to only pick this CSV via environment variable
    env = os.environ.copy()
    env["PYTEST_CURRENT_CSV"] = str(DATA_DIR / csv_file)
    
    cmd = f'python -m pytest tests/testdatacleaning.py --html="{report_name}" --self-contained-html'
    
    subprocess.run(cmd, shell=True, env=env)
