from pathlib import Path
import pandas as pd


# Project root:
# VanPredict/
BASE_DIR = Path(__file__).resolve().parents[3]

# Data folder:
# VanPredict/data/
DATA_DIR = BASE_DIR / "data"

DATA_311 = DATA_DIR / "vanpredict_311_drainage.csv"
DATA_RAIN = DATA_DIR / "vancouver_hourly_rainfall_utc.csv"


def load_311():
    """Load Vancouver 311 drainage complaints."""
    if not DATA_311.exists():
        raise FileNotFoundError(
            f"311 dataset not found at: {DATA_311}"
        )

    return pd.read_csv(DATA_311)


def load_rainfall():
    """Load Vancouver hourly rainfall data."""
    if not DATA_RAIN.exists():
        raise FileNotFoundError(
            f"Rainfall dataset not found at: {DATA_RAIN}"
        )

    return pd.read_csv(DATA_RAIN)