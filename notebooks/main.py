import pandas as pd

# ============================================================
# 1. FILE PATHS
# ============================================================

OLD_FILE = "../data/3-1-1-service-requests-2009-2021.csv"
NEW_FILE = "../data/3-1-1-service-requests.csv"

OUTPUT_FILE = "../data/vanpredict_311_drainage.csv"


# ============================================================
# 2. TARGET SERVICE REQUEST TYPES
# ============================================================

TARGET_TYPES = [
    "Street Surface Water Flooding Case",
    "Sewer Backup Case",
    "Catch Basin Concern Case",
    "Sewer Maintenance Hole Concern Case"
]


# ============================================================
# 3. LOAD BOTH CSV FILES
# ============================================================

print("Loading datasets...")

old_df = pd.read_csv(OLD_FILE, sep=";", low_memory=False)
new_df = pd.read_csv(NEW_FILE, sep=";", low_memory=False)

print("Old dataset:", len(old_df), "records")
print("New dataset:", len(new_df), "records")


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


old_df = clean_columns(old_df)
new_df = clean_columns(new_df)


print("\nOLD COLUMNS:")
print(old_df.columns.tolist())

print("\nNEW COLUMNS:")
print(new_df.columns.tolist())


# ============================================================
# 5. FIND COMMON COLUMNS
# ============================================================

common_columns = list(
    set(old_df.columns).intersection(new_df.columns)
)

print("\nCommon columns:")
print(common_columns)


# Keep only columns that exist in BOTH datasets
old_df = old_df[common_columns].copy()
new_df = new_df[common_columns].copy()


# ============================================================
# 6. COMBINE DATASETS
# ============================================================

df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

print("\nCombined records:", len(df))


# ============================================================
# 7. KEEP ONLY VANPREDICT TARGET EVENTS
# ============================================================

df = df[
    df["service_request_type"].isin(TARGET_TYPES)
].copy()

print("\nRecords after filtering:", len(df))


# ============================================================
# 8. CONVERT DATE
# ============================================================

df["service_request_open_timestamp"] = (
    pd.to_datetime(
        df["service_request_open_timestamp"],
        errors="coerce",
        utc=True
    )
    .dt.tz_convert("America/Vancouver")
)


# ============================================================
# 9. CONVERT COORDINATES TO NUMERIC
# ============================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)


# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicates removed:", before - after)


# ============================================================
# 11. SORT BY DATE
# ============================================================

df = df.sort_values(
    "service_request_open_timestamp"
)


# ============================================================
# 12. SHOW BREAKDOWN
# ============================================================

print("\n========================================")
print("VANPREDICT DATASET")
print("========================================")

print("\nTotal records:", len(df))

print("\nService request breakdown:")

print(
    df["service_request_type"]
    .value_counts()
)


# ============================================================
# 13. DATE RANGE
# ============================================================

print("\nDate range:")

print(
    "Start:",
    df["service_request_open_timestamp"].min()
)

print(
    "End:",
    df["service_request_open_timestamp"].max()
)


# ============================================================
# 14. MISSING COORDINATES
# ============================================================

print("\nMissing coordinates:")

print(
    df[["latitude", "longitude"]]
    .isna()
    .sum()
)


# ============================================================
# 15. MISSING COORDINATES BY TYPE
# ============================================================

print("\nMissing coordinates by service type:")

missing_coordinates = (
    df[
        df[["latitude", "longitude"]]
        .isna()
        .any(axis=1)
    ]
    ["service_request_type"]
    .value_counts()
)

print(missing_coordinates)


# ============================================================
# 16. SAVE FINAL DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("DONE!")
print("========================================")

print(
    f"Saved {len(df)} records to:"
)

print(OUTPUT_FILE)