"""
Baseline forecasts. Every baseline for target week t at horizon h uses only
demand observed up to the origin week t-h (counted in open weeks).

  naive            y(origin)                                  last known week
  moving_avg_4     mean of the 4 weeks ending at the origin   smoothed naive
  seasonal_naive_52  y(t - 52 calendar weeks)                 same week last year
  seasonal_naive_4   y(t - 4*ceil(h/4) open weeks)            "same week last month" proxy
                     (for h = 4 this is identical to naive)

The data covers about 12 months, so seasonal_naive_52 has no history for the
validation and test weeks (Aug-Nov 2010 is not in the dataset). It is implemented
and reported with its coverage so that this limitation is explicit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TARGET = "units_sold"


def baseline_forecasts(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    h = int(horizon)
    df = panel.sort_values(["StockCode", "week_start"]).reset_index(drop=True)
    g = df.groupby("StockCode", sort=False)[TARGET]

    out = df[["StockCode", "week_start", "split", TARGET]].copy()
    out.insert(2, "horizon", h)

    out["naive"] = g.shift(h)
    out["moving_avg_4"] = (g.shift(h).groupby(df["StockCode"], sort=False)
                           .rolling(4, min_periods=4).mean()
                           .reset_index(level=0, drop=True))

    m = 4 * math.ceil(h / 4)
    out["seasonal_naive_4"] = g.shift(m)

    # Same calendar week last year (by date, not by row position)
    last_year = df[["StockCode", "week_start", TARGET]].copy()
    last_year["week_start"] = last_year["week_start"] + pd.Timedelta(weeks=52)
    last_year = last_year.rename(columns={TARGET: "seasonal_naive_52"})
    out = out.merge(last_year, on=["StockCode", "week_start"], how="left")

    return out


BASELINE_COLUMNS = ["naive", "moving_avg_4", "seasonal_naive_4", "seasonal_naive_52"]
