"""
Shared data access for the dashboard (app/dashboard.py) and the API (api/main.py).

Everything is read from files produced by notebooks 01-03; nothing is retrained here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.replenishment import demand_variability, replenishment_plan

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_DIR / "data" / "processed"
REPORTS = PROJECT_DIR / "reports"
MODELS = PROJECT_DIR / "models"

DECLINE_THRESHOLD = 0.5      # recent 4-week mean < 50% of 12-week mean -> flag for review


def _codes(df):
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def load_backtest() -> pd.DataFrame:
    return _codes(pd.read_csv(PROCESSED / "forecasts_backtest.csv", dtype={"StockCode": str},
                              parse_dates=["week_start", "origin_week"]))


def load_future() -> pd.DataFrame:
    return _codes(pd.read_csv(PROCESSED / "forecasts_future.csv", dtype={"StockCode": str},
                              parse_dates=["week_start", "origin_week"]))


def load_history(codes=None) -> pd.DataFrame:
    """Weekly actuals for open, complete weeks (same rules as the models)."""
    p = pd.read_csv(PROCESSED / "weekly_product_demand.csv", dtype={"StockCode": str, "Description": str},
                    parse_dates=["week_start"])
    p = p[~p["store_closed"] & ~p["is_partial_week"]]
    if codes is not None:
        p = p[p["StockCode"].isin(set(codes))]
    return p[["StockCode", "Description", "week_start", "units_sold", "revenue", "avg_unit_price"]]


def load_report(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name, **kw)


def load_model_card() -> dict:
    return json.loads((MODELS / "model_card.json").read_text())


def product_names(future: pd.DataFrame) -> pd.DataFrame:
    return future[["StockCode", "Description"]].drop_duplicates("StockCode").reset_index(drop=True)


def decline_flags(history: pd.DataFrame) -> pd.DataFrame:
    """Recent demand trend at the last observed week; flags possible phase-outs."""
    h = history.sort_values(["StockCode", "week_start"])
    g = h.groupby("StockCode")["units_sold"]
    out = pd.DataFrame({
        "mean_last_4w": g.apply(lambda s: s.tail(4).mean()),
        "mean_last_12w": g.apply(lambda s: s.tail(12).mean()),
    }).reset_index()
    out["trend_4_vs_12"] = out["mean_last_4w"] / out["mean_last_12w"].replace(0, np.nan)
    out["declining_flag"] = out["trend_4_vs_12"] < DECLINE_THRESHOLD
    return out


def variability(backtest: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Per-product sigma of cumulative `weeks`-ahead errors on the validation period."""
    val = backtest[backtest["split"] == "validation"]
    return demand_variability(val, weeks)


def build_plan(backtest, future, lead_time, review_period, service_level, stock=None,
               pack_size=1, min_order_qty=0) -> pd.DataFrame:
    sL = variability(backtest, lead_time)
    sP = variability(backtest, lead_time + review_period)
    return replenishment_plan(future[["StockCode", "horizon", "forecast"]], sL, sP, stock=stock,
                              lead_time=lead_time, review_period=review_period,
                              service_level=service_level, pack_size=pack_size,
                              min_order_qty=min_order_qty)
