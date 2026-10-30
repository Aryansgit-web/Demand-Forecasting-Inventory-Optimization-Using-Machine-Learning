"""
Replenishment logic: periodic-review, order-up-to policy.

For each product, at forecast origin o, with lead time L and review period R:

  lead-time demand      D_L  = sum of forecasts for weeks o+1 .. o+L
  protection demand     D_P  = sum of forecasts for weeks o+1 .. o+L+R
  demand variability    s_L, s_P = RMSE of the cumulative L- and (L+R)-week
                        forecast errors per product, measured out-of-sample
                        on the validation period
  safety stock          SS   = z * s_P,  z = Normal inverse CDF(service level)
  reorder point         ROP  = D_L + z * s_L
  order-up-to level     S    = D_P + SS
  inventory position    IP   = on_hand + on_order - backorders   (USER INPUT)
  suggested order       Q    = max(0, S - IP), rounded up to pack size,
                               raised to MOQ if an order is placed
  stockout risk         P(lead-time demand > IP) ≈ 1 - Phi((IP - D_L) / s_L)

All parameters come from src/config.py and are documented assumptions.
Stock levels are not in the dataset, so any output that needs them is NaN
until on_hand is supplied.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config import (LEAD_TIME_WEEKS, MIN_ORDER_QTY, PACK_SIZE,
                        REVIEW_PERIOD_WEEKS, SERVICE_LEVEL)


def z_value(service_level: float) -> float:
    return NormalDist().inv_cdf(service_level)


# ------------------------------------------------------------ demand variability
def cumulative_forecasts(fc: pd.DataFrame, weeks: int, value_col: str = "forecast",
                         actual_col: str = "units_sold") -> pd.DataFrame:
    """
    From a long table of direct forecasts (StockCode, origin_week, horizon,
    week_start, forecast, units_sold), sum horizons 1..weeks for each origin.
    Origins without all horizons are dropped.
    """
    f = fc[fc["horizon"] <= weeks]
    agg = (f.groupby(["StockCode", "origin_week"])
             .agg(n=("horizon", "nunique"),
                  cum_forecast=(value_col, "sum"),
                  cum_actual=(actual_col, lambda s: s.sum(min_count=len(s))))
             .reset_index())
    agg = agg[agg["n"] == weeks].drop(columns="n")
    agg["weeks"] = weeks
    return agg


def demand_variability(val_fc: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Per-product RMSE of cumulative `weeks`-ahead forecast errors (validation)."""
    c = cumulative_forecasts(val_fc, weeks).dropna(subset=["cum_actual"])
    c["err"] = c["cum_forecast"] - c["cum_actual"]
    out = (c.groupby("StockCode")
             .agg(sigma=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                  n_origins=("err", "size"))
             .reset_index())
    return out.rename(columns={"sigma": f"sigma_{weeks}w"})


# ------------------------------------------------------------ policy
def replenishment_plan(future_fc: pd.DataFrame, sigma_L: pd.DataFrame, sigma_P: pd.DataFrame,
                       stock: pd.DataFrame | None = None,
                       lead_time: int = LEAD_TIME_WEEKS,
                       review_period: int = REVIEW_PERIOD_WEEKS,
                       service_level: float = SERVICE_LEVEL,
                       pack_size: int = PACK_SIZE,
                       min_order_qty: int = MIN_ORDER_QTY) -> pd.DataFrame:
    """
    future_fc : StockCode, horizon, forecast  (forecasts from one origin)
    sigma_L   : StockCode, sigma_{L}w         (from demand_variability(..., L))
    sigma_P   : StockCode, sigma_{L+R}w
    stock     : optional StockCode, on_hand[, on_order, backorders]
    """
    L, P = lead_time, lead_time + review_period
    if P > future_fc["horizon"].max():
        raise ValueError(f"Lead time + review period ({P}) exceeds the forecast horizons available.")
    z = z_value(service_level)

    d = future_fc.pivot_table(index="StockCode", columns="horizon", values="forecast")
    plan = pd.DataFrame({
        "StockCode": d.index,
        "forecast_next_week": d[1].to_numpy(),
        "lead_time_demand": d[list(range(1, L + 1))].sum(axis=1).to_numpy(),
        "protection_demand": d[list(range(1, P + 1))].sum(axis=1).to_numpy(),
    })
    plan = (plan.merge(sigma_L.iloc[:, :2], on="StockCode", how="left")
                .merge(sigma_P.iloc[:, :2], on="StockCode", how="left"))
    sL, sP = plan[f"sigma_{L}w"], plan[f"sigma_{P}w"]

    plan["lead_time_weeks"] = L
    plan["review_period_weeks"] = review_period
    plan["service_level"] = service_level
    plan["safety_stock"] = np.ceil(z * sP)
    plan["reorder_point"] = np.ceil(plan["lead_time_demand"] + z * sL)
    plan["order_up_to_level"] = np.ceil(plan["protection_demand"] + plan["safety_stock"])

    if stock is None:
        plan["on_hand"] = np.nan
        plan["on_order"] = np.nan
        plan["backorders"] = np.nan
    else:
        s = stock.copy()
        for col in ("on_order", "backorders"):
            if col not in s:
                s[col] = 0
        plan = plan.merge(s[["StockCode", "on_hand", "on_order", "backorders"]],
                          on="StockCode", how="left")

    ip = plan["on_hand"] + plan["on_order"].fillna(0) - plan["backorders"].fillna(0)
    plan["inventory_position"] = ip
    raw_q = (plan["order_up_to_level"] - ip).clip(lower=0)
    q = np.ceil(raw_q / pack_size) * pack_size
    q = np.where((q > 0) & (q < min_order_qty), min_order_qty, q)
    plan["suggested_order_qty"] = np.where(ip.isna(), np.nan, q)
    plan["reorder_now"] = np.where(ip.isna(), pd.NA, ip <= plan["reorder_point"])

    with np.errstate(divide="ignore", invalid="ignore"):
        zr = (ip - plan["lead_time_demand"]) / sL
    plan["stockout_risk_lead_time"] = [1 - NormalDist().cdf(v) if np.isfinite(v) else np.nan
                                       for v in zr]
    plan["weeks_of_cover"] = ip / plan["forecast_next_week"].replace(0, np.nan)
    return plan


# ------------------------------------------------------------ backtest
def service_level_backtest(test_fc: pd.DataFrame, sigma_L: pd.DataFrame, lead_time: int,
                           service_level: float = SERVICE_LEVEL) -> pd.DataFrame:
    """
    Check the safety-stock assumption on the test period: for every origin
    where the full lead time lies in the test period, was actual lead-time
    demand <= forecast lead-time demand + z * sigma?  The share of 'covered'
    cycles should be close to the target service level.
    """
    z = z_value(service_level)
    c = cumulative_forecasts(test_fc, lead_time).dropna(subset=["cum_actual"])
    c = c.merge(sigma_L.iloc[:, :2], on="StockCode", how="left")
    c["protected_level"] = c["cum_forecast"] + z * c[f"sigma_{lead_time}w"]
    c["covered"] = c["cum_actual"] <= c["protected_level"]
    c["covered_without_safety_stock"] = c["cum_actual"] <= c["cum_forecast"]
    return c
