
from pathlib import Path
import pandas as pd
import numpy as np

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the raw dataset
csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found inside {RAW_DIR}"
    )

if len(csv_files) > 1:
    raise ValueError(
        "Multiple CSV files found. Select the correct file explicitly."
    )

file_path = csv_files[0]
df = pd.read_csv(file_path)

print("Dataset loaded:", file_path.name)
print("Original shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())

# 2. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("\nStandardized columns:", df.columns.tolist())

# 3. Identify important columns
date_col = "date" if "date" in df.columns else None

# Support both Walmart-style and other common names
sales_candidates = [
    "weekly_sales", "units_sold", "sales",
    "quantity", "demand"
]
sales_col = next(
    (col for col in sales_candidates if col in df.columns),
    None
)

if sales_col is None:
    raise ValueError(
        "Sales column not found. Update sales_candidates "
        "to match your dataset."
    )

key_candidates = [
    "store", "store_id", "dept", "department",
    "product_id", "item_id", "location_id"
]
key_cols = [col for col in key_candidates if col in df.columns]

# 4. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates().copy()
exact_duplicates_removed = before - len(df)

# 5. Convert date values
invalid_dates = 0

if date_col:
    df[date_col] = pd.to_datetime(
        df[date_col], errors="coerce"
    )
    invalid_dates = int(df[date_col].isna().sum())

    # Invalid dates cannot be used for time-series forecasting
    df = df.dropna(subset=[date_col]).copy()

# 6. Convert sales values to numeric
df[sales_col] = pd.to_numeric(
    df[sales_col], errors="coerce"
)

invalid_sales = int(df[sales_col].isna().sum())

# Missing or non-numeric target values cannot be used directly
df = df.dropna(subset=[sales_col]).copy()

# 7. Check missing values in important ID columns
missing_key_rows = int(
    df[key_cols].isna().any(axis=1).sum()
) if key_cols else 0

# Remove rows without required product/store identifiers
if key_cols:
    df = df.dropna(subset=key_cols).copy()

# 8. Flag negative sales instead of deleting them blindly
# Negative values may represent returns or corrections.
df["negative_sales_flag"] = df[sales_col] < 0

negative_sales_count = int(
    df["negative_sales_flag"].sum()
)

# 9. Check duplicate time-series keys
# For Walmart-style data, this is Store + Dept + Date.
series_keys = []

if "date" in df.columns:
    series_keys.append("date")

for col in ["store", "store_id", "dept", "department",
            "product_id", "item_id", "location_id"]:
    if col in df.columns:
        series_keys.append(col)

duplicate_key_rows = 0

if len(series_keys) > 1:
    duplicate_key_rows = int(
        df.duplicated(subset=series_keys, keep=False).sum()
    )

# 10. Save data-quality report
quality_report = pd.DataFrame({
    "check": [
        "original_rows",
        "exact_duplicates_removed",
        "invalid_dates",
        "invalid_or_missing_sales",
        "rows_with_missing_keys",
        "negative_sales_rows",
        "duplicate_time_series_key_rows",
        "cleaned_rows"
    ],
    "value": [
        len(pd.read_csv(file_path)),
        exact_duplicates_removed,
        invalid_dates,
        invalid_sales,
        missing_key_rows,
        negative_sales_count,
        duplicate_key_rows,
        len(df)
    ]
})

quality_report.to_csv(
    INTERIM_DIR / "data_quality_report.csv",
    index=False
)

# 11. Save cleaned interim dataset
output_path = INTERIM_DIR / "sales_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nPreprocessing completed!")
print("Cleaned shape:", df.shape)
print("Saved dataset:", output_path)
print("\nData-quality report:")
print(quality_report.to_string(index=False))
