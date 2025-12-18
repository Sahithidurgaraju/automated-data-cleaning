import pytest
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = Path("data")  # adjust if needed

def clear_csv_log(csv_name: str):
    """
    Deletes only the log file for a specific CSV if it exists.
    """
    logging.shutdown()  # close all handlers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

    # Delete the log file for this CSV only
    csv_log_file = LOG_DIR / f"{csv_name}*.log"
    if csv_log_file.exists():
        csv_log_file.unlink()
        print(f"Cleared log for {csv_name}*")
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f" Log directory created for {csv_name}*")

# ==============================
# Logger factory (per CSV)
# ==============================
def get_logger(csv_name: str) -> logging.Logger:
    """
    Creates or returns a CSV-specific logger.
    Ensures the log directory exists and log file is created automatically.
    """
    # Ensure the log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # unique microseconds
    log_file = LOG_DIR / f"{csv_name}.log"
    logger = logging.getLogger(f"{csv_name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    # Add handler only once
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False

        logger.info(f"Logger initialized for {csv_name}")

    return logger

# ==============================
# Safe logging helper
# ==============================
def safe_log(logger, message):
    """
    Logs anything safely (DataFrame/Series or any object).
    """
    if isinstance(message, (pd.DataFrame, pd.Series)):
        logger.info(message.to_string())
    else:
        logger.info(str(message))
