import pandas as pd


def check_data_quality(df, name):

    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY REPORT: {name}")
    print("=" * 60)

    print("\n1. SHAPE")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\n2. MISSING VALUES")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })

    print(missing[missing["missing_count"] > 0])

    print("\n3. DUPLICATE ROWS")
    print("Duplicates:", df.duplicated().sum())

    print("\n4. DATA TYPES")
    print(df.dtypes)

    print("\n5. SAMPLE")
    print(df.head())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing": df.isna().sum().to_dict()
    }


def check_311_quality(df):

    print("\n\n===== SPECIAL 311 CHECKS =====")

    # Timestamp
    dates = pd.to_datetime(
        df["service_request_open_timestamp"],
        errors="coerce",
        utc=True
    )

    print("Invalid timestamps:", dates.isna().sum())

    # Coordinates
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    print("Missing/invalid latitude:", lat.isna().sum())
    print("Missing/invalid longitude:", lon.isna().sum())

    # Vancouver-ish coordinate sanity check
    invalid_coordinates = (
        (lat < 49.0) |
        (lat > 49.5) |
        (lon < -123.5) |
        (lon > -122.8)
    )

    print(
        "Suspicious coordinates:",
        invalid_coordinates.fillna(True).sum()
    )

    print("\nService request types:")
    print(df["service_request_type"].value_counts().head(30))

    print("\nDate range:")
    print("Start:", dates.min())
    print("End:", dates.max())


def check_rainfall_quality(df):

    print("\n\n===== SPECIAL RAINFALL CHECKS =====")

    dates = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    rain = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce"
    )

    print("Invalid timestamps:", dates.isna().sum())
    print("Invalid rainfall values:", rain.isna().sum())
    print("Negative rainfall:", (rain < 0).sum())

    print("\nRainfall statistics:")
    print(rain.describe())

    print("\nDate range:")
    print("Start:", dates.min())
    print("End:", dates.max())

    print("\nDuplicate timestamps:")
    print(dates.duplicated().sum())