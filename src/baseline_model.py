import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

import joblib


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = "data/vanpredict_311_drainage.csv"
MODEL_FILE = "data/vanpredict_baseline_model.pkl"


# ============================================================
# 2. SETTINGS
# ============================================================

# Approximately 200m x 200m grid
GRID_SIZE = 0.002


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

df = df.dropna(
    subset=["timestamp"]
).copy()

df["date"] = df["timestamp"].dt.date


# ============================================================
# 5. KEEP RECORDS WITH COORDINATES
# ============================================================

df = df.dropna(
    subset=["latitude", "longitude"]
).copy()

print("Records with coordinates:", len(df))


# ============================================================
# 6. CREATE SPATIAL GRID
# ============================================================

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
# 8. AGGREGATE EVENTS BY GRID + DAY
# ============================================================

print("Creating spatial/daily dataset...")

daily = (
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


print("Grid/day rows:", len(daily))


# ============================================================
# 9. CREATE SURFACE DRAINAGE IMPACT
# ============================================================

daily["surface_drainage_impact"] = (
    daily["flooding_reports"]
    +
    daily["catch_basin_reports"]
)


# ============================================================
# 10. SORT DATA
# ============================================================

daily["date"] = pd.to_datetime(
    daily["date"]
)

daily = daily.sort_values(
    [
        "grid_lat",
        "grid_lon",
        "date"
    ]
)


# ============================================================
# 11. CREATE HISTORICAL FEATURES
# ============================================================

# Previous day's flooding
daily["previous_flooding"] = (
    daily
    .groupby(
        ["grid_lat", "grid_lon"]
    )["flooding_reports"]
    .shift(1)
)


# Previous day's catch basin concerns
daily["previous_catch_basin"] = (
    daily
    .groupby(
        ["grid_lat", "grid_lon"]
    )["catch_basin_reports"]
    .shift(1)
)


# Previous day's sewer backups
daily["previous_sewer_backup"] = (
    daily
    .groupby(
        ["grid_lat", "grid_lon"]
    )["sewer_backup_reports"]
    .shift(1)
)


# Previous day's maintenance-hole concerns
daily["previous_maintenance_hole"] = (
    daily
    .groupby(
        ["grid_lat", "grid_lon"]
    )["maintenance_hole_reports"]
    .shift(1)
)


# Replace missing previous-day values with zero
history_columns = [
    "previous_flooding",
    "previous_catch_basin",
    "previous_sewer_backup",
    "previous_maintenance_hole"
]

daily[history_columns] = (
    daily[history_columns]
    .fillna(0)
)


# ============================================================
# 12. SEASONAL FEATURES
# ============================================================

daily["month"] = daily["date"].dt.month

daily["day_of_year"] = (
    daily["date"].dt.dayofyear
)


# ============================================================
# 13. CREATE FUTURE TARGET
# ============================================================

# Tomorrow's surface drainage activity
daily["next_day_impact"] = (
    daily
    .groupby(
        ["grid_lat", "grid_lon"]
    )["surface_drainage_impact"]
    .shift(-1)
)


# ============================================================
# 14. CREATE BINARY TARGET
# ============================================================

# Baseline definition:
# 1 = at least one flooding/catch-basin event tomorrow
# 0 = no such event tomorrow

daily["high_impact_next_day"] = (
    daily["next_day_impact"] > 0
).astype(int)


# Remove rows where tomorrow is unknown
daily = daily.dropna(
    subset=["next_day_impact"]
).copy()


print("\nTarget distribution:")

print(
    daily["high_impact_next_day"]
    .value_counts()
)


# ============================================================
# 15. DEFINE MODEL FEATURES
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

    "month",
    "day_of_year",

    "grid_lat",
    "grid_lon"
]

TARGET = "high_impact_next_day"


X = daily[FEATURES]

y = daily[TARGET]


# ============================================================
# 16. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating time-based train/test split...")

split_date = daily["date"].quantile(0.80)

train_mask = (
    daily["date"] <= split_date
)

test_mask = (
    daily["date"] > split_date
)

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("Split date:", split_date)

print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))


# ============================================================
# 17. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 18. PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)

probabilities = (
    model
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# 19. EVALUATION
# ============================================================

print("\n========================================")
print("BASELINE MODEL RESULTS")
print("========================================")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


if y_test.nunique() == 2:

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        "\nROC-AUC:",
        round(auc, 4)
    )


# ============================================================
# 20. FEATURE IMPORTANCE
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

print(importance.to_string(index=False))


# ============================================================
# 21. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_FILE)