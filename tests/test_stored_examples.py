"""
Reproducibility against stored evaluation outputs from notebook 03:
  * the saved production models re-create the stored future forecasts;
  * the stored evaluation metrics re-compute from the stored backtest forecasts;
  * the model was selected on validation, not on test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import app_data as ad
from src.config import MODEL_HORIZONS
from src.data_cleaning import load_weekly_panel
from src.evaluation import metrics
from src.features import extend_with_future_weeks, make_features
from src.models import MAX_ABS_BIAS_PCT, load_model, predict
from src.splits import model_ready_panel, select_products


@pytest.fixture(scope="module")
def future_features():
    panel = load_weekly_panel(ad.PROCESSED / "weekly_product_demand.csv")
    sel = select_products(panel)
    mp = model_ready_panel(panel)
    mp = mp[mp["StockCode"].isin(sel.loc[sel["selected"], "StockCode"])].reset_index(drop=True)
    mpf = extend_with_future_weeks(mp, max(MODEL_HORIZONS))
    last = mp["week_start"].max()
    out = {}
    for h in MODEL_HORIZONS:
        f = make_features(mpf, h)
        out[h] = f[(f["split"] == "future") & (f["origin_week"] == last)].reset_index(drop=True)
    return out


@pytest.mark.parametrize("h", MODEL_HORIZONS)
@pytest.mark.parametrize("family", ["xgboost", "random_forest"])
def test_saved_models_reproduce_stored_forecasts(future_features, family, h):
    model, meta = load_model(ad.MODELS / f"{family}_h{h}.joblib")
    assert meta["horizon"] == h and meta["model"] == family
    feats = future_features[h]
    pred = predict(model, family, meta["target_mode"], feats, h)
    stored = (ad.load_future().query("horizon == @h").set_index("StockCode")
              .loc[feats["StockCode"], family].to_numpy())
    np.testing.assert_allclose(pred, stored, rtol=1e-5, atol=1e-4)


def test_selected_forecast_is_ensemble_of_components():
    f = ad.load_future()
    card = ad.load_model_card()
    if card["selected_method"] == "ensemble":
        np.testing.assert_allclose(f["forecast"], f[["xgboost", "random_forest", "ses"]].mean(axis=1), rtol=1e-6, atol=1e-3)
    assert (f["forecast"] >= 0).all() and f["forecast"].notna().all()


@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_stored_metrics_recompute(h, split):
    bt = ad.load_backtest()
    comp = ad.load_report("model_comparison.csv")
    sel = ad.load_model_card()["selected_method"]
    rows = bt[(bt["horizon"] == h) & (bt["split"] == split)]
    m = metrics(rows["units_sold"], rows[sel])
    stored = comp[(comp["horizon"] == h) & (comp["split"] == split) & (comp["method"] == sel)].iloc[0]
    assert m["WAPE_%"] == pytest.approx(stored["WAPE_%"], rel=1e-9)
    assert m["MAE"] == pytest.approx(stored["MAE"], rel=1e-9)
    assert m["Bias_%"] == pytest.approx(stored["Bias_%"], rel=1e-9)


def test_model_selected_on_validation_not_test():
    comp = ad.load_report("model_comparison.csv")
    card = ad.load_model_card()
    val = comp[(comp["split"] == "validation") & comp["method"].isin(["xgboost", "random_forest", "ses", "ensemble"])]
    s = val.groupby("method").agg(w=("WAPE_%", "mean"), b=("Bias_%", lambda x: x.abs().max()))
    eligible = s[s["b"] <= MAX_ABS_BIAS_PCT].sort_values("w")
    assert eligible.index[0] == card["selected_method"]


def test_selected_model_beats_main_baseline_on_validation():
    comp = ad.load_report("model_comparison.csv")
    sel = ad.load_model_card()["selected_method"]
    v = comp[comp["split"] == "validation"].pivot_table(index="horizon", columns="method", values="WAPE_%")
    assert (v[sel] < v["moving_avg_4"]).all()
