"""
Reusable data-cleaning functions for the Online Retail demand-forecasting project.

Weeks 2-4 should import from here instead of copying notebook cells:

    from src.data_cleaning import load_clean_sales, load_weekly_panel

Cleaning rules (documented in README / reports/cleaning_log.csv):
  1. Exact duplicate rows are removed.
  2. Non-product stock codes (postage, fees, manual adjustments, gift vouchers)
     are removed. Products are codes that start with 5 digits.
  3. Orders that were later cancelled in full (same customer, product and
     quantity, cancellation on or after the order) are removed together
     with the cancellation.
  4. Remaining rows with Quantity <= 0 (other cancellations / adjustments)
     or UnitPrice <= 0 are removed. Demand = gross units ordered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Codes that start with 5 digits are real products (e.g. 85123A, 71053, 15056BL).
PRODUCT_CODE_PATTERN = r"^\d{5}"

CLEAN_SALES_DTYPES = {
    "InvoiceNo": str,
    "StockCode": str,
    "Description": str,
    "Quantity": "int64",
    "UnitPrice": "float64",
    "CustomerID": "float64",
    "Country": str,
    "Revenue": "float64",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw Excel file and fix column types. Never edits the file."""
    df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _match_full_cancellations(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """
    Pair each cancellation row with the most recent earlier order of the
    same customer, product and quantity. Returns (order_index, cancel_index)
    of matched rows. Each order and each cancellation is used at most once.
    """
    is_cancel = df["InvoiceNo"].str.startswith("C") & (df["Quantity"] < 0)
    cancels = df[is_cancel & df["CustomerID"].notna()].copy()
    orders = df[(~is_cancel) & (df["Quantity"] > 0) & df["CustomerID"].notna()].copy()

    cancels["match_qty"] = -cancels["Quantity"]
    orders["match_qty"] = orders["Quantity"]

    keys = ["CustomerID", "StockCode", "match_qty"]
    pairs = (
        cancels[keys + ["InvoiceDate"]].reset_index(names="cancel_idx")
        .merge(
            orders[keys + ["InvoiceDate"]].reset_index(names="order_idx"),
            on=keys,
            suffixes=("_cancel", "_order"),
        )
    )
    pairs = pairs[pairs["InvoiceDate_order"] <= pairs["InvoiceDate_cancel"]]

    # Latest qualifying order per cancellation, then one cancellation per order.
    pairs = (
        pairs.sort_values("InvoiceDate_order", ascending=False)
        .drop_duplicates("cancel_idx")
        .drop_duplicates("order_idx")
    )
    return pd.Index(pairs["order_idx"]), pd.Index(pairs["cancel_idx"])


def clean_sales(
    df: pd.DataFrame,
    countries: list[str] | None = None,
    return_log: bool = False,
):
    """
    Apply the documented cleaning rules and return transaction-level sales.

    Parameters
    ----------
    df : raw dataframe from load_raw()
    countries : optional list such as ["United Kingdom"] to restrict scope
    return_log : if True, also return a dataframe of rows removed per step
    """
    log = []

    def step(name, before, after, note):
        log.append({"Step": name, "Rows before": before, "Rows removed": before - after,
                    "Rows after": after, "Rule": note})

    out = df.copy()

    n = len(out)
    out = out[out["InvoiceDate"].notna()]
    step("Invalid dates", n, len(out), "InvoiceDate could not be parsed")

    n = len(out)
    out = out.drop_duplicates()
    step("Exact duplicates", n, len(out), "All 8 columns identical")

    n = len(out)
    out = out[out["StockCode"].str.match(PRODUCT_CODE_PATTERN)]
    step("Non-product codes", n, len(out),
         "POST, DOT, M, D, C2, S, BANK CHARGES, AMAZONFEE, CRUK, PADS, B, gift vouchers, DCGS*")

    n = len(out)
    order_idx, cancel_idx = _match_full_cancellations(out)
    out = out.drop(index=order_idx.union(cancel_idx))
    step("Fully cancelled orders", n, len(out),
         "Order + matching cancellation removed (same customer, product, quantity)")

    n = len(out)
    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    step("Non-positive qty / price", n, len(out),
         "Remaining cancellations, returns, adjustments and zero-price rows")

    if countries is not None:
        n = len(out)
        out = out[out["Country"].isin(countries)]
        step("Country scope", n, len(out), f"Kept only {countries}")

    out = out.copy()
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    out = out.sort_values("InvoiceDate").reset_index(drop=True)

    if return_log:
        return out, pd.DataFrame(log)
    return out


def save_clean_sales(sales: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(path, index=False)


def load_clean_sales(path: str | Path) -> pd.DataFrame:
    """Read cleaned_sales.csv back with correct types (StockCode stays text)."""
    return pd.read_csv(path, dtype=CLEAN_SALES_DTYPES, parse_dates=["InvoiceDate"])


# --------------------------------------------------------------------------
# Calendar / weekly product panel
# --------------------------------------------------------------------------
def trading_calendar(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Daily calendar over the full date range with a flag for days the shop
    traded. Days with no transactions at all are treated as closed (missing),
    not as zero demand.
    """
    days = pd.date_range(sales["InvoiceDate"].dt.normalize().min(),
                         sales["InvoiceDate"].dt.normalize().max(), freq="D")
    traded = set(sales["InvoiceDate"].dt.normalize().unique())
    cal = pd.DataFrame({"date": days})
    cal["weekday"] = cal["date"].dt.day_name()
    cal["is_trading_day"] = cal["date"].isin(traded)
    return cal


def build_weekly_panel(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Complete product x week table (weeks start Monday).

    - Each product appears from its first sale week to the last week in the data,
      so weeks before a product was launched are NOT filled with zeros.
    - Weeks inside that range with no sales get units_sold = 0 (true zero),
      unless the shop was closed all week (store_closed = True).
    - trading_days and is_partial_week let Week 2 drop or adjust incomplete weeks.
    """
    s = sales.copy()
    s["week_start"] = s["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

    cal = trading_calendar(s)
    cal["week_start"] = cal["date"].dt.to_period("W-SUN").dt.start_time
    weeks = cal.groupby("week_start").agg(
        trading_days=("is_trading_day", "sum"),
        calendar_days_in_data=("date", "size"),
    ).reset_index()
    weeks["store_closed"] = weeks["trading_days"] == 0
    weeks["is_partial_week"] = weeks["calendar_days_in_data"] < 7

    weekly = (
        s.groupby(["StockCode", "week_start"])
        .agg(units_sold=("Quantity", "sum"),
             revenue=("Revenue", "sum"),
             n_orders=("InvoiceNo", "nunique"),
             avg_unit_price=("UnitPrice", "mean"))
        .reset_index()
    )

    first_week = weekly.groupby("StockCode")["week_start"].min()
    all_weeks = weeks["week_start"].sort_values().to_numpy()
    grid = pd.DataFrame(
        [(code, w) for code, fw in first_week.items() for w in all_weeks if w >= fw],
        columns=["StockCode", "week_start"],
    )

    panel = (grid.merge(weekly, on=["StockCode", "week_start"], how="left")
                 .merge(weeks, on="week_start", how="left"))
    panel["units_sold"] = panel["units_sold"].fillna(0).astype("int64")
    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel["n_orders"] = panel["n_orders"].fillna(0).astype("int64")
    # avg_unit_price stays NaN in zero-sales weeks (no price observed).

    names = (s.groupby("StockCode")["Description"]
               .agg(lambda x: x.mode().iat[0] if not x.mode().empty else pd.NA))
    panel["Description"] = panel["StockCode"].map(names)

    cols = ["StockCode", "Description", "week_start", "units_sold", "revenue",
            "n_orders", "avg_unit_price", "trading_days", "store_closed", "is_partial_week"]
    return panel[cols].sort_values(["StockCode", "week_start"]).reset_index(drop=True)


def load_weekly_panel(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"StockCode": str, "Description": str},
                       parse_dates=["week_start"])


# --------------------------------------------------------------------------
# Demand profiling (intermittency)
# --------------------------------------------------------------------------
def demand_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-product demand statistics using only complete, open weeks.

    ADI  = average interval between weeks with demand (1 = sells every week)
    CV2  = squared coefficient of variation of non-zero weekly demand
    Class (Syntetos-Boylan): smooth / erratic / intermittent / lumpy
    """
    p = panel[~panel["store_closed"] & ~panel["is_partial_week"]]

    def stats(g):
        nz = g.loc[g["units_sold"] > 0, "units_sold"]
        n_weeks = len(g)
        n_nz = len(nz)
        adi = n_weeks / n_nz if n_nz else np.inf
        cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if n_nz > 1 else np.nan
        return pd.Series({
            "total_units": g["units_sold"].sum(),
            "weeks_observed": n_weeks,
            "weeks_with_sales": n_nz,
            "pct_weeks_with_sales": n_nz / n_weeks if n_weeks else np.nan,
            "ADI": adi,
            "CV2": cv2,
        })

    prof = p.groupby("StockCode")[["units_sold"]].apply(stats).reset_index()
    prof["demand_class"] = np.select(
        [(prof["ADI"] < 1.32) & (prof["CV2"] < 0.49),
         (prof["ADI"] < 1.32) & (prof["CV2"] >= 0.49),
         (prof["ADI"] >= 1.32) & (prof["CV2"] < 0.49),
         (prof["ADI"] >= 1.32) & (prof["CV2"] >= 0.49)],
        ["smooth", "erratic", "intermittent", "lumpy"],
        default="insufficient data",
    )
    return prof.sort_values("total_units", ascending=False).reset_index(drop=True)
