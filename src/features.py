"""
Leakage-safe feature engineering for weekly product demand.

Direct multi-horizon setup: for a forecast made at origin week o for target
week t = o + h, every demand-derived feature only uses weeks <= o, i.e. it is
shifted by at least h open weeks. Calendar features describe the target week
itself, which is allowed because the shop calendar and public holidays are
known in advance.

    from src.features import make_features
    feats = make_features(model_panel, horizon=1)

Shifts are counted in *open weeks* (closed / partial weeks were removed),
so lag 1 of the week of 3 Jan 2011 is the week of 20 Dec 2010.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# England & Wales bank holidays inside the data range (gov.uk).
UK_BANK_HOLIDAYS = pd.to_datetime([
    "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25",
    "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26", "2011-12-27",
])

BASE_LAGS = (1, 2, 3, 4, 8, 12)        # relative to the forecast origin
ROLL_WINDOWS = (4, 8, 12)
TARGET = "units_sold"


def _lag_list(horizon: int) -> list[int]:
    """Lags measured from the target week: origin lag L -> shift L + h - 1."""
    return [L + horizon - 1 for L in BASE_LAGS]


def calendar_features(week_start: pd.Series) -> pd.DataFrame:
    """Features of the target week that are known in advance."""
    w = pd.to_datetime(week_start)
    week_end = w + pd.Timedelta(days=6)
    hol = UK_BANK_HOLIDAYS.values
    n_hol = [int(((hol >= s) & (hol <= e)).sum()) for s, e in zip(w.values, week_end.values)]
    xmas = pd.to_datetime(w.dt.year.astype(str) + "-12-25")
    return pd.DataFrame({
        "week_of_year": w.dt.isocalendar().week.astype(int).values,
        "month": w.dt.month.values,
        "quarter": w.dt.quarter.values,
        "n_bank_holidays": n_hol,
        "weeks_to_christmas": ((xmas - w).dt.days // 7).values,
    }, index=week_start.index)


def _weeks_since_last_sale(s: pd.Series) -> pd.Series:
    """Open weeks since the last week with sales, evaluated on an already-shifted series."""
    out = np.full(len(s), np.nan)
    last = None
    for i, v in enumerate(s.to_numpy()):
        if not np.isnan(v):
            if v > 0:
                last = i
            out[i] = (i - last) if last is not None else np.nan
    return pd.Series(out, index=s.index)


def make_features(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Build the model table for one forecast horizon.

    Input: model-ready panel (closed / partial weeks removed) with columns
    StockCode, week_start, units_sold, n_orders, avg_unit_price, trading_days, split.
    Output: one row per product-week with target, features and metadata.
    """
    h = int(horizon)
    df = panel.sort_values(["StockCode", "week_start"]).reset_index(drop=True).copy()
    g = df.groupby("StockCode", sort=False)

    # Demand as known at the forecast origin (shifted h open weeks)
    y_origin = g[TARGET].shift(h)
    gy = y_origin.groupby(df["StockCode"], sort=False)

    feats = {}
    for k in _lag_list(h):
        feats[f"lag_{k}"] = g[TARGET].shift(k)

    for w in ROLL_WINDOWS:
        roll = gy.rolling(w, min_periods=max(2, w // 2))
        feats[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        feats[f"roll_std_{w}"] = roll.std().reset_index(level=0, drop=True)
    feats["roll_max_8"] = gy.rolling(8, min_periods=4).max().reset_index(level=0, drop=True)
    feats["share_weeks_with_sales_8"] = ((y_origin > 0).astype(float).where(y_origin.notna())
                                         .groupby(df["StockCode"], sort=False)
                                         .rolling(8, min_periods=4).mean()
                                         .reset_index(level=0, drop=True))
    feats["ewm_mean"] = gy.transform(lambda s: s.ewm(alpha=0.3, adjust=False, ignore_na=True).mean())
    feats["weeks_since_last_sale"] = gy.transform(_weeks_since_last_sale)
    feats["trend_4_vs_12"] = feats["roll_mean_4"] / feats["roll_mean_12"].replace(0, np.nan)

    # Orders and price as known at the origin
    feats["orders_lag_h"] = g["n_orders"].shift(h)
    feats["orders_roll_mean_4"] = (feats["orders_lag_h"].groupby(df["StockCode"], sort=False)
                                   .rolling(4, min_periods=2).mean()
                                   .reset_index(level=0, drop=True))
    price_origin = g["avg_unit_price"].shift(h).groupby(df["StockCode"], sort=False).ffill()
    feats["price_last_known"] = price_origin
    price_ref = (price_origin.groupby(df["StockCode"], sort=False)
                 .rolling(8, min_periods=2).mean().reset_index(level=0, drop=True))
    feats["price_vs_8wk_mean"] = price_origin / price_ref

    # Product age at the target week (known: weeks since first sale in the panel)
    feats["product_age_weeks"] = g.cumcount()

    out = pd.concat([df[["StockCode", "Description", "week_start", "split", TARGET, "trading_days"]],
                     pd.DataFrame(feats, index=df.index),
                     calendar_features(df["week_start"])], axis=1)
    out.insert(3, "horizon", h)
    out["origin_week"] = g["week_start"].shift(h)
    out["has_full_history"] = out[[f"lag_{k}" for k in _lag_list(h)]].notna().all(axis=1)
    return out


def load_features(path) -> pd.DataFrame:
    """Read features_h*.csv with StockCode kept as text and dates parsed."""
    return pd.read_csv(path, dtype={"StockCode": str, "Description": str},
                       parse_dates=["week_start", "origin_week"])


def feature_dictionary(horizon: int = 1) -> pd.DataFrame:
    h = horizon
    rows = [(f"lag_{k}", "Units sold k open weeks before the target week", f"origin (t-{h}) or earlier")
            for k in _lag_list(h)]
    for w in ROLL_WINDOWS:
        rows += [(f"roll_mean_{w}", f"Mean weekly units over the {w} weeks ending at the origin", "origin"),
                 (f"roll_std_{w}", f"Std of weekly units over the {w} weeks ending at the origin", "origin")]
    rows += [
        ("roll_max_8", "Max weekly units over 8 weeks ending at the origin", "origin"),
        ("share_weeks_with_sales_8", "Share of the last 8 weeks with any sales (intermittency)", "origin"),
        ("ewm_mean", "Exponentially weighted mean of units (alpha 0.3)", "origin"),
        ("weeks_since_last_sale", "Open weeks since the most recent week with sales", "origin"),
        ("trend_4_vs_12", "roll_mean_4 / roll_mean_12 (recent momentum)", "origin"),
        ("orders_lag_h", "Number of invoices containing the product at the origin week", "origin"),
        ("orders_roll_mean_4", "Mean invoices per week over 4 weeks ending at the origin", "origin"),
        ("price_last_known", "Last observed average unit price (forward-filled)", "origin"),
        ("price_vs_8wk_mean", "Last price / 8-week mean price (price-change proxy; no promo data)", "origin"),
        ("product_age_weeks", "Open weeks since the product's first sale", "calendar (known)"),
        ("trading_days", "Shop trading days in the target week", "calendar (known)"),
        ("week_of_year", "ISO week number of the target week", "calendar (known)"),
        ("month", "Month of the target week", "calendar (known)"),
        ("quarter", "Quarter of the target week", "calendar (known)"),
        ("n_bank_holidays", "England & Wales bank holidays in the target week", "calendar (known)"),
        ("weeks_to_christmas", "Weeks from target week start to 25 Dec of the same year", "calendar (known)"),
    ]
    return pd.DataFrame(rows, columns=["feature", "description", "uses data up to"]).assign(horizon=h)


def feature_columns(horizon: int = 1) -> list[str]:
    return feature_dictionary(horizon)["feature"].tolist()


def check_no_leakage(panel: pd.DataFrame, horizon: int, cutoff: pd.Timestamp,
                     seed: int = 0) -> bool:
    """
    Scramble every target value after `cutoff` and rebuild the features.
    Features for target weeks up to `horizon` open weeks after the cutoff must
    be identical, otherwise a feature is looking into the future.
    """
    rng = np.random.default_rng(seed)
    base = make_features(panel, horizon)
    scrambled = panel.copy()
    future = scrambled["week_start"] > cutoff
    for col in ["units_sold", "n_orders", "avg_unit_price"]:
        scrambled.loc[future, col] = rng.permutation(scrambled.loc[future, col].to_numpy())
    test = make_features(scrambled, horizon)

    weeks = np.sort(panel["week_start"].unique())
    pos = np.searchsorted(weeks, np.datetime64(cutoff), side="right") - 1
    last_safe = weeks[min(pos + horizon, len(weeks) - 1)]
    rows = base["week_start"] <= last_safe
    cols = feature_columns(horizon)
    a, b = base.loc[rows, cols], test.loc[rows, cols]
    same = ((a == b) | (a.isna() & b.isna())).all().all()
    return bool(same)
