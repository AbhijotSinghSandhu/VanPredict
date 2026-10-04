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
MODEL_PATH = ROOT / "data" / "vanpredict_baseline_v04_model.pkl"


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
print("VANPREDICT v0.4")
print("Same-day rainfall → drainage prediction")
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


# ============================================================
# DEFINE DRAINAGE IMPACT
# ============================================================

# We are focusing on surface drainage problems:
#
#   Street Surface Water Flooding
#   Catch Basin Concern
#
# These are combined into the target.

daily_events["surface_drainage_impact"] = (
    daily_events["flooding_reports"]
    + daily_events["catch_basin_reports"]
)


# ============================================================
# CREATE SPATIAL GRID
# ============================================================

print("\nCreating complete spatial grid...")

grid_cells = daily_events[
    ["grid_lat", "grid_lon"]
].drop_duplicates()

print(
    f"Unique grid cells: {len(grid_cells):,}"
)


# ============================================================
# DATE RANGE
# ============================================================

min_date = df["date"].min()
max_date = df["date"].max()

dates = pd.date_range(
    start=min_date,
    end=max_date,
    freq="D",
    tz="America/Vancouver"
)

print("\nDate range:")
print(f"Start: {min_date}")
print(f"End:   {max_date}")
print(f"Calendar days: {len(dates):,}")


# ============================================================
# CREATE COMPLETE GRID × DATE DATASET
# ============================================================

print("\nCreating full grid × calendar-day dataset...")

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
# HISTORICAL DRAINAGE FEATURES
# ============================================================

print("\nCreating historical drainage features...")

grouped = panel.groupby(
    ["grid_lat", "grid_lon"],
    sort=False
)


# ------------------------------------------------------------
# PREVIOUS DAY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# PREVIOUS 7 DAYS
# ------------------------------------------------------------

panel["flooding_7d"] = (
    grouped["flooding_reports"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7, min_periods=1)
        .sum()
    )
)

panel["catch_basin_7d"] = (
    grouped["catch_basin_reports"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7, min_periods=1)
        .sum()
    )
)

panel["maintenance_hole_7d"] = (
    grouped["maintenance_hole_reports"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7, min_periods=1)
        .sum()
    )
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

# Remove accidental Markdown asterisks if present.
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
        # Rainfall occurring on THIS day
        rainfall_today=(
            "rainfall_mm",
            "sum"
        ),

        # Maximum intensity / accumulation
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

        # Heavy rainfall indicators
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


# ============================================================
# PREVIOUS-DAY / ANTECEDENT RAINFALL
# ============================================================

# Previous day rainfall.
# Does NOT include today's rainfall.

daily_rain["rainfall_previous_day"] = (
    daily_rain["rainfall_today"]
    .shift(1)
)


# Previous 3 days.
# Shift first so today's rainfall is excluded.

daily_rain["rainfall_previous_3d"] = (
    daily_rain["rainfall_today"]
    .shift(1)
    .rolling(3, min_periods=1)
    .sum()
)


# Previous 7 days.
# Again, today's rainfall is excluded.

daily_rain["rainfall_previous_7d"] = (
    daily_rain["rainfall_today"]
    .shift(1)
    .rolling(7, min_periods=1)
    .sum()
)


# ============================================================
# RAINFALL DURATION
# ============================================================

# Count how many hourly observations had measurable rain.
#
# This gives the Random Forest a rough measure of how
# long rainfall persisted during the day.

rain["rain_present"] = (
    rain["rainfall_mm"] > 0
).astype(int)

rain_duration = (
    rain.groupby("date_key")["rain_present"]
    .sum()
    .rename("rain_duration_hours")
)

daily_rain = daily_rain.join(
    rain_duration,
    how="left"
)


# ============================================================
# RAINFALL FEATURES
# ============================================================

rainfall_features = [
    "rainfall_today",
    "rainfall_previous_day",
    "rainfall_previous_3d",
    "rainfall_previous_7d",
    "max_rain_1h",
    "max_rain_6h",
    "max_rain_24h",
    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",
    "rain_duration_hours",
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


# Missing rainfall records are treated as zero.
for col in rainfall_features:
    panel[col] = panel[col].fillna(0)


# ============================================================
# CREATE SAME-DAY TARGET
# ============================================================

print("\nCreating same-day drainage target...")

# IMPORTANT:
#
# We are NOT predicting tomorrow anymore.
#
# The target is whether a drainage problem happened
# on THIS grid cell on THIS day.

panel["is_drainage_problem"] = (
    panel["surface_drainage_impact"] > 0
).astype(int)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    # Location
    "grid_lat",
    "grid_lon",

    # Historical drainage context
    "previous_flooding",
    "previous_catch_basin",
    "previous_sewer_backup",
    "previous_maintenance_hole",

    "flooding_7d",
    "catch_basin_7d",
    "maintenance_hole_7d",

    # Calendar
    "month",
    "day_of_year",
    "day_of_week",

    # Current-day rainfall
    "rainfall_today",
    "max_rain_1h",
    "max_rain_6h",
    "max_rain_24h",
    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",
    "rain_duration_hours",

    # Rainfall BEFORE today
    "rainfall_previous_day",
    "rainfall_previous_3d",
    "rainfall_previous_7d",
]


# ============================================================
# REMOVE DATA THAT WOULD LEAK THE TARGET
# ============================================================

# We intentionally DO NOT put these into feature_columns:
#
# flooding_reports
# catch_basin_reports
# maintenance_hole_reports
# sewer_backup_reports
# surface_drainage_impact
#
# because they describe what happened on the target day.
#
# The model must only see information available to it
# when making the prediction.


# ============================================================
# CLEAN NaN VALUES
# ============================================================

panel[feature_columns] = (
    panel[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n========================================")
print("VANPREDICT v0.4 DATASET")
print("========================================")

print(
    f"Grid/day rows: {len(panel):,}"
)

positive_count = (
    panel["is_drainage_problem"].sum()
)

negative_count = (
    panel["is_drainage_problem"] == 0
).sum()

print(
    f"Positive same-day cases: "
    f"{positive_count:,}"
)

print(
    f"Negative same-day cases: "
    f"{negative_count:,}"
)

print(
    f"Positive rate: "
    f"{positive_count / len(panel) * 100:.3f}%"
)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
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
    feature_columns + ["is_drainage_problem"]
].copy()

test_data = panel.loc[
    test_mask,
    feature_columns + ["is_drainage_problem"]
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
    train_data["is_drainage_problem"] == 1
]

negative_data = train_data[
    train_data["is_drainage_problem"] == 0
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
# PREPARE TRAINING DATA
# ============================================================

X_train = balanced_train[
    feature_columns
]

y_train = balanced_train[
    "is_drainage_problem"
]

X_test = test_data[
    feature_columns
]

y_test = test_data[
    "is_drainage_problem"
]


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

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
print("VANPREDICT v0.4 RESULTS")
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

print(
    importance.to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "grid_size": GRID_SIZE,
        "version": "v0.4",
        "target": "is_drainage_problem",
    },
    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_PATH)