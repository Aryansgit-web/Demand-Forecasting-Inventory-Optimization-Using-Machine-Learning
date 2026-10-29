"""
Chronological split plan, forecast horizons and modelling scope.

All dates are week_start values (Monday). Weeks flagged store_closed or
is_partial_week in the Week 1 panel are removed before modelling.

    Train       2010-12-06 .. 2011-08-08   35 open weeks (week of 27 Dec 2010 closed)
    Validation  2011-08-15 .. 2011-10-03    8 weeks  (tuning / model choice)
    Test        2011-10-10 .. 2011-11-28    8 weeks  (final evaluation only)
"""

from __future__ import annotations

import pandas as pd

TRAIN_START = pd.Timestamp("2010-12-06")
TRAIN_END = pd.Timestamp("2011-08-08")
VALID_START = pd.Timestamp("2011-08-15")
VALID_END = pd.Timestamp("2011-10-03")
TEST_START = pd.Timestamp("2011-10-10")
TEST_END = pd.Timestamp("2011-11-28")

# h = 1: next-week operational forecast.  h = 4: about one month ahead, used for
# replenishment planning in Week 3 (covers an assumed lead time of up to 4 weeks).
HORIZONS = (1, 4)

# Product scope rule (computed on TRAIN weeks only, so no test information is used)
MIN_SHARE_WEEKS_WITH_SALES = 0.80


def assign_split(week_start: pd.Series) -> pd.Series:
    """Label each week as train / validation / test / outside."""
    w = pd.to_datetime(week_start)
    out = pd.Series("outside", index=w.index, dtype="object")
    out[(w >= TRAIN_START) & (w <= TRAIN_END)] = "train"
    out[(w >= VALID_START) & (w <= VALID_END)] = "validation"
    out[(w >= TEST_START) & (w <= TEST_END)] = "test"
    return out


def model_ready_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop closed and partial weeks and add the split label."""
    p = panel[~panel["store_closed"] & ~panel["is_partial_week"]].copy()
    p["split"] = assign_split(p["week_start"])
    p = p[p["split"] != "outside"]
    return p.sort_values(["StockCode", "week_start"]).reset_index(drop=True)


def select_products(panel: pd.DataFrame,
                    min_share: float = MIN_SHARE_WEEKS_WITH_SALES) -> pd.DataFrame:
    """
    Choose products that can be forecast at weekly product level.

    Rule (train weeks only):
      * present in every train week, i.e. first sold before the train period started
      * sold in at least `min_share` of train weeks
    Returns one row per product with the statistics used for the decision.
    """
    p = model_ready_panel(panel)
    train = p[p["split"] == "train"]
    n_train_weeks = train["week_start"].nunique()

    stats = (train.groupby("StockCode")
                  .agg(train_weeks_present=("week_start", "nunique"),
                       train_weeks_with_sales=("units_sold", lambda s: (s > 0).sum()),
                       train_units=("units_sold", "sum"))
                  .reset_index())
    stats["share_weeks_with_sales"] = stats["train_weeks_with_sales"] / n_train_weeks
    stats["full_train_history"] = stats["train_weeks_present"] == n_train_weeks
    stats["selected"] = stats["full_train_history"] & (stats["share_weeks_with_sales"] >= min_share)
    return stats.sort_values("train_units", ascending=False).reset_index(drop=True)


def split_plan_table(panel: pd.DataFrame) -> pd.DataFrame:
    p = model_ready_panel(panel)
    rows = []
    notes = {
        "train": "Fit models and baselines. Includes Dec-2010 peak and the Christmas closure gap.",
        "validation": "Tune hyper-parameters and choose models. Start of the autumn ramp-up.",
        "test": "Final evaluation only, never used for tuning. Peak pre-Christmas season.",
    }
    for name, (s, e) in {"train": (TRAIN_START, TRAIN_END),
                         "validation": (VALID_START, VALID_END),
                         "test": (TEST_START, TEST_END)}.items():
        weeks = p.loc[p["split"] == name, "week_start"].drop_duplicates()
        rows.append({"split": name, "start_week": s.date(), "end_week": e.date(),
                     "open_weeks": len(weeks), "notes": notes[name]})
    excluded = panel.loc[panel["store_closed"] | panel["is_partial_week"], "week_start"].drop_duplicates()
    rows.append({"split": "excluded", "start_week": None, "end_week": None,
                 "open_weeks": len(excluded),
                 "notes": "Removed: " + ", ".join(str(d.date()) for d in sorted(excluded))
                          + " (closed or partial weeks)"})
    return pd.DataFrame(rows)
