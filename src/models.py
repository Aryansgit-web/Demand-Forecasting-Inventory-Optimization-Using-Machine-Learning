"""
Forecasting models: XGBoost, Random Forest, simple exponential smoothing (SES)
and an equal-weight ensemble of the three.

Workflow (per horizon h, one direct model per horizon):
  1. tune_models():   fit candidates on TRAIN, score on VALIDATION.
  2. select_best():   per model family, lowest validation WAPE among candidates
                      whose |bias| <= MAX_ABS_BIAS_PCT (an unbiased forecast is
                      needed for replenishment; median-type forecasts such as
                      log targets or L1 loss score well on WAPE but run ~30-40%
                      low and would cause stockouts).
  3. fit_final():     refit the chosen configurations on TRAIN + VALIDATION,
                      predict TEST once for the final comparison.
  4. production fit:  refit on ALL observed weeks to forecast the future.

Target options
  raw     predict units (squared error, or Tweedie loss for zero-heavy counts)
  log1p   predict log(1 + units)                       (rejected: biased low)
  ratio   predict units / recent 8-week mean, weighted by that mean
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_SEED
from src.evaluation import metrics
from src.features import feature_columns

TARGET = "units_sold"
MAX_ABS_BIAS_PCT = 10.0

_XGB_BASE = {"learning_rate": 0.05, "n_estimators": 400, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 10}
_RF_BASE = {"n_estimators": 200, "max_features": 0.5, "max_depth": 16}

# (family, target_mode, params) candidates evaluated on validation
CANDIDATES = (
    [("xgboost", "raw", {**_XGB_BASE, "max_depth": d}) for d in (3, 5)]
    + [("xgboost", "log1p", {**_XGB_BASE, "max_depth": 3})]
    + [("xgboost", "raw", {**_XGB_BASE, "max_depth": d, "objective": "reg:tweedie",
                           "tweedie_variance_power": vp})
       for vp in (1.3, 1.5) for d in (3, 5)]
    + [("random_forest", mode, {**_RF_BASE, "min_samples_leaf": leaf})
       for mode in ("raw", "ratio") for leaf in (20, 50)]
    + [("random_forest", "log1p", {**_RF_BASE, "min_samples_leaf": 20})]
)
SES_ALPHAS = (0.1, 0.2, 0.3, 0.5)


# ------------------------------------------------------------------ data prep
def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting: target known and full lag history."""
    return df[df["has_full_history"] & df[TARGET].notna()]


def _scale(df: pd.DataFrame) -> np.ndarray:
    s = df["roll_mean_8"].fillna(df["ewm_mean"]).fillna(0).to_numpy(dtype=float)
    return np.maximum(s, 1.0)


def make_xy(df: pd.DataFrame, horizon: int, mode: str):
    X = df[feature_columns(horizon)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    scale = _scale(df)
    weight = None
    if mode == "log1p":
        y = np.log1p(y)
    elif mode == "ratio":
        y = y / scale
        weight = scale
    return X, y, scale, weight


def back_transform(pred: np.ndarray, scale: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        pred = np.expm1(pred)
    elif mode == "ratio":
        pred = pred * scale
    return np.clip(pred, 0, None)


# ------------------------------------------------------------------ models
def build_model(kind: str, params: dict):
    if kind == "xgboost":
        return XGBRegressor(tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
                            **{"objective": "reg:squarederror", **params})
    if kind == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **params)
    raise ValueError(kind)


def _X(kind, X):
    return np.nan_to_num(X, nan=-1.0) if kind == "random_forest" else X


def fit_model(kind: str, params: dict, mode: str, train: pd.DataFrame, horizon: int):
    X, y, _, w = make_xy(train, horizon, mode)
    model = build_model(kind, params)
    model.fit(_X(kind, X), y, sample_weight=w)
    return model


def predict(model, kind: str, mode: str, df: pd.DataFrame, horizon: int) -> np.ndarray:
    X, _, scale, _ = make_xy(df, horizon, mode)
    return back_transform(model.predict(_X(kind, X)), scale, mode)


def ses_forecast(panel: pd.DataFrame, horizon: int, alpha: float) -> pd.Series:
    """SES level at the origin (t-h), used as a flat forecast for week t."""
    df = panel.sort_values(["StockCode", "week_start"])
    y_origin = df.groupby("StockCode", sort=False)[TARGET].shift(horizon)
    level = (y_origin.groupby(df["StockCode"], sort=False)
             .transform(lambda s: s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()))
    return level.reindex(panel.index)


def add_ses(feats: pd.DataFrame, panel: pd.DataFrame, horizon: int, alpha: float) -> np.ndarray:
    p = panel[["StockCode", "week_start"]].copy()
    p["ses"] = ses_forecast(panel, horizon, alpha)
    return feats[["StockCode", "week_start"]].merge(p, on=["StockCode", "week_start"],
                                                    how="left")["ses"].to_numpy()


# ------------------------------------------------------------------ tuning
def tune_models(feats: pd.DataFrame, panel: pd.DataFrame, horizon: int,
                verbose: bool = True) -> pd.DataFrame:
    """Fit each candidate on train, score on validation."""
    train = training_rows(feats[feats["split"] == "train"])
    valid = feats[feats["split"] == "validation"]
    rows = []
    for kind, mode, params in CANDIDATES:
        model = fit_model(kind, params, mode, train, horizon)
        m = metrics(valid[TARGET], predict(model, kind, mode, valid, horizon))
        rows.append({"horizon": horizon, "model": kind, "target_mode": mode,
                     "params": json.dumps(params), **m})
        if verbose:
            print(f"h={horizon} {kind:13s} {mode:5s} WAPE={m['WAPE_%']:6.2f}  Bias={m['Bias_%']:7.2f}")
    for a in SES_ALPHAS:
        m = metrics(valid[TARGET], add_ses(valid, panel, horizon, a))
        rows.append({"horizon": horizon, "model": "ses", "target_mode": "raw",
                     "params": json.dumps({"alpha": a}), **m})
    out = pd.DataFrame(rows)
    out["bias_ok"] = out["Bias_%"].abs() <= MAX_ABS_BIAS_PCT
    return out


def select_best(tuning: pd.DataFrame) -> pd.DataFrame:
    """Best configuration per (horizon, family): lowest WAPE with acceptable bias."""
    ok = tuning[tuning["bias_ok"]]
    return (ok.sort_values("WAPE_%").groupby(["horizon", "model"], as_index=False).head(1)
              .sort_values(["horizon", "WAPE_%"]).reset_index(drop=True))


# ------------------------------------------------------------------ persistence
def save_model(model, path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path: Path):
    path = Path(path)
    return joblib.load(path), json.loads(path.with_suffix(".json").read_text())


def feature_importance(model, horizon: int) -> pd.DataFrame:
    return (pd.DataFrame({"feature": feature_columns(horizon),
                          "importance": model.feature_importances_})
              .sort_values("importance", ascending=False).reset_index(drop=True))
