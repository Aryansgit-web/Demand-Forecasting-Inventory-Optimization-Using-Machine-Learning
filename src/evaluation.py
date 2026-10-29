"""
Forecast evaluation metrics.

  MAE        mean |forecast - actual|                    units
  RMSE       sqrt(mean (forecast - actual)^2)            units, penalises big misses
  WAPE       sum |error| / sum actual                    % of volume; safe with zeros
  Bias %     sum (forecast - actual) / sum actual        + = over-forecast, - = under
  MAPE       mean |error| / actual, only where actual > 0  shown for reference only
  coverage   share of rows where the method could produce a forecast

All methods are compared on the SAME rows (rows where every compared method
has a forecast), so the comparison is fair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual, forecast) -> dict:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    e = f - a
    nz = a > 0
    total = a.sum()
    return {
        "n": len(a),
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "WAPE_%": 100 * np.abs(e).sum() / total if total else np.nan,
        "Bias_%": 100 * e.sum() / total if total else np.nan,
        "MAPE_%_nonzero": 100 * np.mean(np.abs(e[nz]) / a[nz]) if nz.any() else np.nan,
    }


def evaluation_table(df: pd.DataFrame, methods: list[str], actual: str = "units_sold",
                     by: list[str] | None = None, common_rows: bool = True) -> pd.DataFrame:
    """
    Metrics per method (and optionally per group such as split / horizon).
    Methods with zero coverage on a group are reported with coverage only.
    """
    by = by or []
    rows = []
    groups = df.groupby(by) if by else [((), df)]
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        usable = [m for m in methods if part[m].notna().any()]
        mask = part[usable].notna().all(axis=1) if (common_rows and usable) else None
        for m in methods:
            cov = part[m].notna().mean()
            base = {**dict(zip(by, key)), "method": m, "coverage_%": 100 * cov}
            if cov == 0:
                rows.append(base)
                continue
            sel = mask if mask is not None else part[m].notna()
            rows.append({**base, **metrics(part.loc[sel, actual], part.loc[sel, m])})
    return pd.DataFrame(rows)


def error_by_product(df: pd.DataFrame, method: str, actual: str = "units_sold") -> pd.DataFrame:
    d = df[df[method].notna()]
    return (d.groupby("StockCode")
             .apply(lambda g: pd.Series(metrics(g[actual], g[method])), include_groups=False)
             .reset_index()
             .sort_values("MAE", ascending=False))
