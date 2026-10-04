from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DRAINAGE_PATH = ROOT / "data" / "vanpredict_311_drainage.csv"
RAINFALL_PATH = ROOT / "data" / "vancouver_hourly_rainfall_2009_2026.csv"
MODEL_PATH = ROOT / "data" / "vanpredict_baseline_v03_model.pkl"


# ============================================================
# SETTINGS
# ============================================================

GRID_SIZE = 0.002
RANDOM_STATE = 42

NEGATIVE_RATIO = 3

TEST_START = "2023-04-19"


# ============================================================
# LOAD 311 DATA
# ============================================================

print("========================================")
print("VANPREDICT v0.3")
print("Rainfall-aware drainage prediction")
print("========================================")

print("\nLoading VANPREDICT drainage data...")

df = pd.read_csv(DRAINAGE_PATH)

print(f"Total records: {len(df):,}")


# ============================================================
# PARSE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["service_request_open_timestamp"],
    errors="coerce",
    utc=True
)

df["timestamp"] = df["timestamp"].dt.tz_convert(
    "America/Vancouver"
)

df["date"] = df["timestamp"].dt.normalize()


# ============================================================
# COORDINATES
# ============================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "latitude", "longitude"]
).copy()

print(
    f"Records with coordinates: {len(df):,}"
)


# ============================================================
# CREATE SPATIAL GRID
# ============================================================

print("\nCreating spatial grid...")

df["grid_lat"] = (
    np.floor(df["latitude"] / GRID_SIZE)
    * GRID_SIZE
)

df["grid_lon"] = (
    np.floor(df["longitude"] / GRID_SIZE)
    * GRID_SIZE
)


# ============================================================
# EVENT FLAGS
# ============================================================

df["flooding_reports"] = (
    df["service_request_type"]
    == "Street Surface Water Flooding Case"
).astype(int)

df["catch_basin_reports"] = (
    df["service_request_type"]
    == "Catch Basin Concern Case"
).astype(int)

df["maintenance_hole_reports"] = (
    df["service_request_type"]
    == "Sewer Maintenance Hole Concern Case"
).astype(int)

df["sewer_backup_reports"] = (
    df["service_request_type"]
    == "Sewer Backup Case"
).astype(int)


# ============================================================
# AGGREGATE 311 EVENTS BY GRID + DAY
# ============================================================

print("\nAggregating drainage events...")

daily_events = (
    df.groupby(
        ["date", "grid_lat", "grid_lon"]
    )
    [
        [
            "flooding_reports",
            "catch_basin_reports",
            "maintenance_hole_reports",
            "sewer_backup_reports",
        ]
    ]
    .sum()
    .reset_index()
)


daily_events["surface_drainage_impact"] = (
    daily_events["flooding_reports"]
    + daily_events["catch_basin_reports"]
)


# ============================================================
# CREATE COMPLETE GRID × DATE DATASET
# ============================================================

print("\nCreating complete spatial grid...")

grid_cells = daily_events[
    ["grid_lat", "grid_lon"]
].drop_duplicates()

print(
    f"Unique grid cells: {len(grid_cells):,}"
)


min_date = df["date"].min()
max_date = df["date"].max()

dates = pd.date_range(
    start=min_date,
    end=max_date,
    freq="D",
    tz="America/Vancouver"
)

print("Date range:")
print(f"Start: {min_date}")
print(f"End:   {max_date}")
print(f"Calendar days: {len(dates):,}")


print("\nCreating full grid × calendar-day dataset...")

# Create grid × date combinations
grid_cells["_key"] = 1

date_df = pd.DataFrame({
    "date": dates,
    "_key": 1
})

panel = grid_cells.merge(
    date_df,
    on="_key"
).drop(columns="_key")


print(
    f"Potential grid/day combinations: "
    f"{len(panel):,}"
)


# ============================================================
# MERGE EVENTS
# ============================================================

panel = panel.merge(
    daily_events,
    on=["date", "grid_lat", "grid_lon"],
    how="left"
)


event_columns = [
    "flooding_reports",
    "catch_basin_reports",
    "maintenance_hole_reports",
    "sewer_backup_reports",
    "surface_drainage_impact",
]

for col in event_columns:
    panel[col] = panel[col].fillna(0)


# ============================================================
# SORT
# ============================================================

panel = panel.sort_values(
    ["grid_lat", "grid_lon", "date"]
).reset_index(drop=True)


# ============================================================
# HISTORICAL ACTIVITY FEATURES
# ============================================================

print("\nCreating historical drainage features...")

grouped = panel.groupby(
    ["grid_lat", "grid_lon"],
    sort=False
)

panel["previous_flooding"] = (
    grouped["flooding_reports"].shift(1)
)

panel["previous_catch_basin"] = (
    grouped["catch_basin_reports"].shift(1)
)

panel["previous_sewer_backup"] = (
    grouped["sewer_backup_reports"].shift(1)
)

panel["previous_maintenance_hole"] = (
    grouped["maintenance_hole_reports"].shift(1)
)


# 7-day rolling activity
panel["flooding_7d"] = (
    grouped["flooding_reports"]
    .rolling(7, min_periods=1)
    .sum()
    .reset_index(level=[0, 1], drop=True)
)

panel["catch_basin_7d"] = (
    grouped["catch_basin_reports"]
    .rolling(7, min_periods=1)
    .sum()
    .reset_index(level=[0, 1], drop=True)
)

panel["maintenance_hole_7d"] = (
    grouped["maintenance_hole_reports"]
    .rolling(7, min_periods=1)
    .sum()
    .reset_index(level=[0, 1], drop=True)
)


# ============================================================
# CALENDAR FEATURES
# ============================================================

print("Creating calendar features...")

panel["month"] = panel["date"].dt.month

panel["day_of_year"] = (
    panel["date"].dt.dayofyear
)

panel["day_of_week"] = (
    panel["date"].dt.dayofweek
)


# ============================================================
# LOAD RAINFALL
# ============================================================

print("\nLoading rainfall data...")

rain = pd.read_csv(RAINFALL_PATH)

# Remove accidental Markdown asterisks if present
rain.columns = (
    rain.columns
    .str.strip()
    .str.replace("*", "", regex=False)
)

print(
    f"Rainfall rows: {len(rain):,}"
)


# ============================================================
# PARSE RAINFALL
# ============================================================

rain["datetime"] = pd.to_datetime(
    rain["datetime"],
    errors="coerce"
)

rain = rain.dropna(
    subset=["datetime"]
).copy()


rainfall_columns = [
    "rainfall_mm",
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
    "rain_48h",
    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",
]

for col in rainfall_columns:
    rain[col] = pd.to_numeric(
        rain[col],
        errors="coerce"
    )


# ============================================================
# RAINFALL DATE
# ============================================================

rain["date_key"] = (
    rain["datetime"].dt.normalize()
)


# ============================================================
# DAILY RAINFALL FEATURES
# ============================================================

print("\nCreating daily rainfall features...")

daily_rain = (
    rain.groupby("date_key")
    .agg(
        rainfall_today=(
            "rainfall_mm",
            "sum"
        ),

        max_rain_1h=(
            "rain_1h",
            "max"
        ),

        max_rain_6h=(
            "rain_6h",
            "max"
        ),

        max_rain_24h=(
            "rain_24h",
            "max"
        ),

        heavy_rain_1h=(
            "heavy_rain_1h",
            "max"
        ),

        heavy_rain_6h=(
            "heavy_rain_6h",
            "max"
        ),

        heavy_rain_24h=(
            "heavy_rain_24h",
            "max"
        ),
    )
    .sort_index()
)


# Historical rainfall
daily_rain["rainfall_previous_day"] = (
    daily_rain["rainfall_today"].shift(1)
)

daily_rain["rainfall_3d"] = (
    daily_rain["rainfall_today"]
    .rolling(3, min_periods=1)
    .sum()
)

daily_rain["rainfall_7d"] = (
    daily_rain["rainfall_today"]
    .rolling(7, min_periods=1)
    .sum()
)


rainfall_features = [
    "rainfall_today",
    "rainfall_previous_day",
    "rainfall_3d",
    "rainfall_7d",
    "max_rain_1h",
    "max_rain_6h",
    "max_rain_24h",
    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",
]

daily_rain = daily_rain[
    rainfall_features
].reset_index()


print(
    f"Daily rainfall rows: "
    f"{len(daily_rain):,}"
)


# ============================================================
# MERGE RAINFALL INTO PANEL
# ============================================================

print("\nMerging rainfall into VANPREDICT panel...")

# Convert panel timezone-aware date into
# timezone-naive calendar date for merging.

panel["date_key"] = (
    panel["date"]
    .dt.tz_localize(None)
    .dt.normalize()
)

panel = panel.merge(
    daily_rain,
    on="date_key",
    how="left"
)

panel = panel.drop(
    columns=["date_key"]
)


# Rainfall is city-wide, so missing dates mean
# no rainfall record for that day.
for col in rainfall_features:
    panel[col] = panel[col].fillna(0)


# ============================================================
# CREATE NEXT-DAY TARGET
# ============================================================

print("\nCreating next-calendar-day target...")

grouped = panel.groupby(
    ["grid_lat", "grid_lon"],
    sort=False
)

panel["next_day_impact"] = (
    grouped["surface_drainage_impact"]
    .shift(-1)
)

panel = panel.dropna(
    subset=["next_day_impact"]
).copy()

panel["high_impact_next_day"] = (
    panel["next_day_impact"] > 0
).astype(int)


# ============================================================
# CLEAN NaN VALUES
# ============================================================

feature_columns = [
    "grid_lat",
    "grid_lon",

    "flooding_reports",
    "catch_basin_reports",
    "maintenance_hole_reports",
    "sewer_backup_reports",

    "previous_flooding",
    "previous_catch_basin",
    "previous_sewer_backup",
    "previous_maintenance_hole",

    "flooding_7d",
    "catch_basin_7d",
    "maintenance_hole_7d",

    "month",
    "day_of_year",
    "day_of_week",

    # Rainfall
    "rainfall_today",
    "rainfall_previous_day",
    "rainfall_3d",
    "rainfall_7d",
    "max_rain_1h",
    "max_rain_6h",
    "max_rain_24h",
    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",
]

panel[feature_columns] = (
    panel[feature_columns]
    .fillna(0)
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n========================================")
print("VANPREDICT v0.3 DATASET")
print("========================================")

print(
    f"Grid/day rows: {len(panel):,}"
)

print(
    f"Positive next-day cases: "
    f"{panel['high_impact_next_day'].sum():,}"
)

print(
    f"Negative next-day cases: "
    f"{(panel['high_impact_next_day'] == 0).sum():,}"
)


# ============================================================
# TIME SPLIT
# ============================================================

print("\nCreating time-based train/test split...")

split_date = pd.Timestamp(
    TEST_START,
    tz="America/Vancouver"
)

train_mask = (
    panel["date"] < split_date
)

test_mask = (
    panel["date"] >= split_date
)

train_data = panel.loc[
    train_mask,
    feature_columns + ["high_impact_next_day"]
].copy()

test_data = panel.loc[
    test_mask,
    feature_columns + ["high_impact_next_day"]
].copy()


print(
    f"Training rows: {len(train_data):,}"
)

print(
    f"Testing rows: {len(test_data):,}"
)


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print("\nBalancing training data...")

positive_data = train_data[
    train_data["high_impact_next_day"] == 1
]

negative_data = train_data[
    train_data["high_impact_next_day"] == 0
]

print(
    f"Positive training examples: "
    f"{len(positive_data):,}"
)

print(
    f"Negative training examples: "
    f"{len(negative_data):,}"
)


negative_sample_size = min(
    len(negative_data),
    len(positive_data) * NEGATIVE_RATIO
)

negative_sample = negative_data.sample(
    n=negative_sample_size,
    random_state=RANDOM_STATE
)


balanced_train = pd.concat(
    [
        positive_data,
        negative_sample
    ],
    ignore_index=True
)

balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nBalanced training rows: "
    f"{len(balanced_train):,}"
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

X_train = balanced_train[
    feature_columns
]

y_train = balanced_train[
    "high_impact_next_day"
]

X_test = test_data[
    feature_columns
]

y_test = test_data[
    "high_impact_next_day"
]


print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

print("\n========================================")
print("VANPREDICT v0.3 RESULTS")
print("========================================")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")

print(importance.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "grid_size": GRID_SIZE,
        "version": "v0.3",
    },
    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_PATH)