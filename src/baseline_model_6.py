from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DRAINAGE_PATH = ROOT / "data" / "vanpredict_311_drainage.csv"
RAINFALL_PATH = ROOT / "data" / "vancouver_hourly_rainfall_2009_2026.csv"
MODEL_PATH = ROOT / "data" / "vanpredict_baseline_v06_model.pkl"


# ============================================================
# SETTINGS
# ============================================================

GRID_SIZE = 0.002
RANDOM_STATE = 42

NEGATIVE_RATIO = 3

# Only grid/day observations with at least this much
# rainfall are included in the v0.6 experiment.
RAINY_DAY_THRESHOLD = 1.0

TEST_START = "2023-04-19"


# ============================================================
# LOAD 311 DATA
# ============================================================

print("========================================")
print("VANPREDICT v0.6")
print("Rainy-day rainfall + location → drainage")
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

# Surface drainage problems:
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
# Today's rainfall is excluded.

daily_rain["rainfall_previous_day"] = (
    daily_rain["rainfall_today"]
    .shift(1)
)


# Previous 3 days.
# Today's rainfall is excluded.

daily_rain["rainfall_previous_3d"] = (
    daily_rain["rainfall_today"]
    .shift(1)
    .rolling(3, min_periods=1)
    .sum()
)


# Previous 7 days.
# Today's rainfall is excluded.

daily_rain["rainfall_previous_7d"] = (
    daily_rain["rainfall_today"]
    .shift(1)
    .rolling(7, min_periods=1)
    .sum()
)


# ============================================================
# RAINFALL DURATION
# ============================================================

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

panel["is_drainage_problem"] = (
    panel["surface_drainage_impact"] > 0
).astype(int)


# ============================================================
# FILTER TO RAINY GRID/DAY OBSERVATIONS
# ============================================================

print("\nFiltering to rainy grid/day observations...")

rainy_panel = panel[
    panel["rainfall_today"] >= RAINY_DAY_THRESHOLD
].copy()

print(
    f"Rainy grid/day rows: "
    f"{len(rainy_panel):,}"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

# V0.6 intentionally removes historical drainage features.
#
# The purpose of this experiment is to answer:
#
#   "Given rainfall and location, can we identify
#    locations that are more likely to experience
#    a drainage problem?"
#
# Historical drainage features can be added later
# as a separate experiment.

feature_columns = [

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    "grid_lat",
    "grid_lon",

    # --------------------------------------------------------
    # Calendar / seasonality
    # --------------------------------------------------------

    "month",
    "day_of_year",
    "day_of_week",

    # --------------------------------------------------------
    # Current-day rainfall
    # --------------------------------------------------------

    "rainfall_today",
    "max_rain_1h",
    "max_rain_6h",
    "max_rain_24h",

    "heavy_rain_1h",
    "heavy_rain_6h",
    "heavy_rain_24h",

    "rain_duration_hours",

    # --------------------------------------------------------
    # Rainfall BEFORE today
    # --------------------------------------------------------

    "rainfall_previous_day",
    "rainfall_previous_3d",
    "rainfall_previous_7d",
]


# ============================================================
# IMPORTANT:
# DO NOT INCLUDE TARGET-DAY DRAINAGE COUNTS
# ============================================================

# These columns describe what actually happened today:
#
# flooding_reports
# catch_basin_reports
# maintenance_hole_reports
# sewer_backup_reports
# surface_drainage_impact
#
# They are NOT included in feature_columns.


# ============================================================
# CLEAN NaN VALUES
# ============================================================

rainy_panel[feature_columns] = (
    rainy_panel[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n========================================")
print("VANPREDICT v0.6 DATASET")
print("========================================")

print(
    f"Rainy grid/day rows: "
    f"{len(rainy_panel):,}"
)

positive_count = (
    rainy_panel["is_drainage_problem"].sum()
)

negative_count = (
    rainy_panel["is_drainage_problem"] == 0
).sum()

print(
    f"Positive rainy grid/day cases: "
    f"{positive_count:,}"
)

print(
    f"Negative rainy grid/day cases: "
    f"{negative_count:,}"
)

print(
    f"Positive rate among rainy observations: "
    f"{positive_count / len(rainy_panel) * 100:.3f}%"
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
    rainy_panel["date"] < split_date
)

test_mask = (
    rainy_panel["date"] >= split_date
)


train_data = rainy_panel.loc[
    train_mask,
    feature_columns + ["is_drainage_problem"]
].copy()

test_data = rainy_panel.loc[
    test_mask,
    feature_columns + ["is_drainage_problem"]
].copy()


print(
    f"Training rainy grid/day rows: "
    f"{len(train_data):,}"
)

print(
    f"Testing rainy grid/day rows: "
    f"{len(test_data):,}"
)


# ============================================================
# BALANCE TRAINING DATA
#
# BOTH CLASSES ARE NOW RAINY.
#
# Positive:
#   rainy day + drainage problem
#
# Negative:
#   rainy day + no drainage problem
#
# This makes the model focus on:
#
#   rainfall + location -> drainage response
# ============================================================

print("\nBalancing rainy-day training data...")

positive_data = train_data[
    train_data["is_drainage_problem"] == 1
].copy()

negative_data = train_data[
    train_data["is_drainage_problem"] == 0
].copy()


print(
    f"Positive rainy training examples: "
    f"{len(positive_data):,}"
)

print(
    f"Negative rainy training examples: "
    f"{len(negative_data):,}"
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

negative_sample_size = min(
    len(negative_data),
    len(positive_data) * NEGATIVE_RATIO
)

negative_sample = negative_data.sample(
    n=negative_sample_size,
    random_state=RANDOM_STATE
)


# ============================================================
# CREATE BALANCED TRAINING DATASET
# ============================================================

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

print(
    "Positive rows: "
    f"{(balanced_train['is_drainage_problem'] == 1).sum():,}"
)

print(
    "Negative rows: "
    f"{(balanced_train['is_drainage_problem'] == 0).sum():,}"
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
# PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("VANPREDICT v0.6 RESULTS")
print("========================================")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print(
    f"PR-AUC:  {pr_auc:.4f}"
)


# ============================================================
# TOP-K RISK PERFORMANCE
# ============================================================

print("\n========================================")
print("TOP-K RISK PERFORMANCE")
print("========================================")

results = test_data[
    ["is_drainage_problem"]
].copy()

results["risk_probability"] = probabilities

results = results.sort_values(
    "risk_probability",
    ascending=False
).reset_index(drop=True)


total_test_problems = (
    y_test.sum()
)


for k in [25, 50, 100, 250, 500]:

    if k > len(results):
        continue

    top_k = results.head(k)

    actual_problems = (
        top_k["is_drainage_problem"].sum()
    )

    precision_at_k = (
        actual_problems / k
    )

    if total_test_problems > 0:

        recall_at_k = (
            actual_problems
            / total_test_problems
        )

    else:

        recall_at_k = 0


    print(
        f"\nTop {k} grid/day predictions:"
    )

    print(
        f"  Actual drainage problems found: "
        f"{actual_problems}"
    )

    print(
        f"  Precision@{k}: "
        f"{precision_at_k:.4f}"
    )

    print(
        f"  Recall@{k}: "
        f"{recall_at_k:.4f}"
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
        "rainy_day_threshold": RAINY_DAY_THRESHOLD,
        "version": "v0.6",
        "target": "is_drainage_problem",
        "training_population": "rainy_grid_days_only",
    },
    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_PATH)

print("\nDone.")