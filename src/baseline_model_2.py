import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# ============================================================
# VANPREDICT BASELINE v0.2
# Full spatial grid × calendar-day dataset
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "vanpredict_311_drainage.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "vanpredict_baseline_v02_model.pkl"


# ============================================================
# 2. SETTINGS
# ============================================================

# Approximately 200 m × 200 m at Vancouver's latitude.
# This is a BASELINE grid size, not our final choice.
GRID_SIZE = 0.002

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading VANPREDICT drainage data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Total records:", len(df))


# ============================================================
# 4. CONVERT TIMESTAMP
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


# Remove rows where timestamp could not be parsed

df = df.dropna(
    subset=["timestamp", "date"]
).copy()


# ============================================================
# 5. CONVERT COORDINATES
# ============================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)


# Keep only spatially usable records

df = df.dropna(
    subset=["latitude", "longitude"]
).copy()

print("Records with coordinates:", len(df))


# ============================================================
# 6. CREATE SPATIAL GRID
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
# 7. CREATE EVENT FLAGS
# ============================================================

df["flooding_reports"] = (
    df["service_request_type"]
    == "Street Surface Water Flooding Case"
).astype(int)

df["catch_basin_reports"] = (
    df["service_request_type"]
    == "Catch Basin Concern Case"
).astype(int)

df["sewer_backup_reports"] = (
    df["service_request_type"]
    == "Sewer Backup Case"
).astype(int)

df["maintenance_hole_reports"] = (
    df["service_request_type"]
    == "Sewer Maintenance Hole Concern Case"
).astype(int)


# ============================================================
# 8. AGGREGATE REAL EVENTS BY GRID + DAY
# ============================================================

daily_events = (
    df.groupby(
        [
            "date",
            "grid_lat",
            "grid_lon"
        ]
    )
    .agg(
        flooding_reports=(
            "flooding_reports",
            "sum"
        ),

        catch_basin_reports=(
            "catch_basin_reports",
            "sum"
        ),

        sewer_backup_reports=(
            "sewer_backup_reports",
            "sum"
        ),

        maintenance_hole_reports=(
            "maintenance_hole_reports",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# 9. CREATE COMPLETE GRID
# ============================================================

print("Creating complete spatial grid...")

grid_cells = (
    df[
        [
            "grid_lat",
            "grid_lon"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Unique grid cells:", len(grid_cells))


# ============================================================
# 10. CREATE COMPLETE CALENDAR
# ============================================================

min_date = df["date"].min()
max_date = df["date"].max()

print("Date range:")
print("Start:", min_date)
print("End:", max_date)

all_dates = pd.date_range(
    start=min_date,
    end=max_date,
    freq="D",
    tz="America/Vancouver"
)

print("Calendar days:", len(all_dates))


# ============================================================
# 11. CREATE GRID × DAY PANEL
# ============================================================

print("\nCreating full grid × calendar-day dataset...")

grid_cells["key"] = 1

calendar = pd.DataFrame({
    "date": all_dates
})

calendar["key"] = 1


panel = grid_cells.merge(
    calendar,
    on="key"
).drop(
    columns=["key"]
)


print(
    "Potential grid/day combinations:",
    len(panel)
)


# ============================================================
# 12. MERGE ACTUAL EVENTS INTO COMPLETE PANEL
# ============================================================

panel = panel.merge(
    daily_events,
    on=[
        "date",
        "grid_lat",
        "grid_lon"
    ],
    how="left"
)


# Any grid/day without a 311 event becomes ZERO

event_columns = [
    "flooding_reports",
    "catch_basin_reports",
    "sewer_backup_reports",
    "maintenance_hole_reports"
]

panel[event_columns] = (
    panel[event_columns]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 13. CREATE COMBINED SURFACE DRAINAGE SIGNAL
# ============================================================

panel["surface_drainage_impact"] = (
    panel["flooding_reports"]
    + panel["catch_basin_reports"]
)


# ============================================================
# 14. SORT DATA
# ============================================================

panel = panel.sort_values(
    [
        "grid_lat",
        "grid_lon",
        "date"
    ]
).reset_index(drop=True)


# ============================================================
# 15. CREATE HISTORICAL LAG FEATURES
# ============================================================

print("Creating historical lag features...")

grouped = panel.groupby(
    [
        "grid_lat",
        "grid_lon"
    ]
)

panel["previous_flooding"] = (
    grouped["flooding_reports"]
    .shift(1)
    .fillna(0)
)

panel["previous_catch_basin"] = (
    grouped["catch_basin_reports"]
    .shift(1)
    .fillna(0)
)

panel["previous_sewer_backup"] = (
    grouped["sewer_backup_reports"]
    .shift(1)
    .fillna(0)
)

panel["previous_maintenance_hole"] = (
    grouped["maintenance_hole_reports"]
    .shift(1)
    .fillna(0)
)


# ============================================================
# 16. ADD 7-DAY HISTORICAL ACTIVITY
# ============================================================

panel["flooding_7d"] = (
    grouped["flooding_reports"]
    .transform(
        lambda x: x.shift(1).rolling(
            7,
            min_periods=1
        ).sum()
    )
)

panel["catch_basin_7d"] = (
    grouped["catch_basin_reports"]
    .transform(
        lambda x: x.shift(1).rolling(
            7,
            min_periods=1
        ).sum()
    )
)

panel["maintenance_hole_7d"] = (
    grouped["maintenance_hole_reports"]
    .transform(
        lambda x: x.shift(1).rolling(
            7,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 17. CALENDAR FEATURES
# ============================================================

panel["month"] = (
    panel["date"].dt.month
)

panel["day_of_year"] = (
    panel["date"].dt.dayofyear
)

panel["day_of_week"] = (
    panel["date"].dt.dayofweek
)


# ============================================================
# 18. CREATE TRUE NEXT-CALENDAR-DAY TARGET
# ============================================================

print("Creating next-calendar-day target...")

# Because EVERY grid/day combination now exists,
# shift(-1) means exactly TOMORROW for each grid cell.

panel["next_day_impact"] = (
    grouped["surface_drainage_impact"]
    .shift(-1)
)


# Remove final day because tomorrow is outside our dataset

panel = panel.dropna(
    subset=["next_day_impact"]
).copy()


# ============================================================
# 19. CREATE CLASSIFICATION TARGET
# ============================================================

# Baseline definition:
# Any surface drainage event tomorrow = positive.
#
# This is intentionally simple for v0.2.
# We will improve the definition after evaluating the baseline.

panel["high_impact_next_day"] = (
    panel["next_day_impact"] > 0
).astype(int)


# ============================================================
# 20. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n========================================")
print("VANPREDICT v0.2 DATASET")
print("========================================")

print(
    "Grid/day rows:",
    len(panel)
)

print(
    "Positive next-day cases:",
    panel["high_impact_next_day"].sum()
)

print(
    "Negative next-day cases:",
    (panel["high_impact_next_day"] == 0).sum()
)

print("\nTarget distribution:")

print(
    panel["high_impact_next_day"]
    .value_counts()
)


# ============================================================
# 21. TRAINING FEATURES
# ============================================================

FEATURES = [
    "flooding_reports",
    "catch_basin_reports",
    "sewer_backup_reports",
    "maintenance_hole_reports",

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

    "grid_lat",
    "grid_lon"
]


X = panel[FEATURES]

y = panel["high_impact_next_day"]


# ============================================================
# 22. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating time-based train/test split...")

split_date = (
    panel["date"]
    .quantile(0.80)
)

print(
    "Split date:",
    split_date
)

train_mask = (
    panel["date"] < split_date
)

test_mask = (
    panel["date"] >= split_date
)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 23. BALANCE TRAINING DATA
# ============================================================

print("\nBalancing training data...")

train_data = panel.loc[
    train_mask,
    FEATURES + ["high_impact_next_day"]
].copy()

positive_data = train_data[
    train_data["high_impact_next_day"] == 1
]

negative_data = train_data[
    train_data["high_impact_next_day"] == 0
]

print(
    "Positive training examples:",
    len(positive_data)
)

print(
    "Negative training examples:",
    len(negative_data)
)


# Use 3 negative examples for every positive example

NEGATIVE_RATIO = 3

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


# Shuffle the training data

balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


X_train = balanced_train[FEATURES]

y_train = balanced_train["high_impact_next_day"]


print(
    "\nBalanced training rows:",
    len(balanced_train)
)

print("\nBalanced target distribution:")

print(
    y_train.value_counts()
)


# ============================================================
# 24. TRAIN RANDOM FOREST
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
# 24. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 25. EVALUATION
# ============================================================

print("\n========================================")
print("BASELINE v0.2 RESULTS")
print("========================================")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    "\nROC-AUC:",
    round(roc_auc, 4)
)


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 27. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "grid_size": GRID_SIZE
    },
    MODEL_FILE
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    MODEL_FILE
)


# ============================================================
# 28. SAVE PANEL FOR FUTURE PIPELINE
# ============================================================

PANEL_FILE = PROJECT_ROOT / "data" / "vanpredict_v02_panel.csv"

panel.to_csv(
    PANEL_FILE,
    index=False
)

print("\nFull v0.2 panel saved to:")

print(
    PANEL_FILE
)

print("\nVANPREDICT v0.2 complete.")

