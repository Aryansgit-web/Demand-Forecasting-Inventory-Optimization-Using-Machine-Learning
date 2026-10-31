"""API behaviour: outputs match the stored forecast files; invalid input gives clear errors."""

import pandas as pd
import pytest

from src import app_data as ad

FUTURE = ad.load_future()
# Stored evaluation examples: the 5 highest-volume products + 5 random ones, all horizons
_codes = FUTURE.groupby("StockCode")["forecast"].sum().sort_values(ascending=False)
EXAMPLE_CODES = list(_codes.index[:5]) + list(_codes.sample(5, random_state=0).index)
EXAMPLES = FUTURE[FUTURE["StockCode"].isin(EXAMPLE_CODES)]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["products"] == FUTURE["StockCode"].nunique()
    assert body["horizons"] == [1, 2, 3, 4]


def test_products_search(client):
    r = client.get("/products", params={"search": "heart", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert 0 < len(body["products"]) <= 5
    assert all("HEART" in (p["description"] or "").upper() or "HEART" in p["stock_code"]
               for p in body["products"])


@pytest.mark.parametrize("row", EXAMPLES.to_dict("records"),
                         ids=lambda r: f"{r['StockCode']}-h{r['horizon']}")
def test_forecast_matches_stored_output(client, row):
    r = client.post("/forecast", json={"stock_code": row["StockCode"], "horizon": int(row["horizon"])})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["forecast_units"] == pytest.approx(row["forecast"], abs=0.01)
    assert body["target_week"] == str(row["week_start"].date())
    assert body["components"]["xgboost"] == pytest.approx(row["xgboost"], abs=0.01)
    assert len(body["path"]) == row["horizon"]
    lo, hi = body["interval_90"]["lower"], body["interval_90"]["upper"]
    assert lo <= body["forecast_units"] <= hi
    assert body["forecast_units"] >= 0


def test_get_equals_post_and_case_insensitive(client):
    code = EXAMPLE_CODES[0]
    post = client.post("/forecast", json={"stock_code": code.lower(), "horizon": 3}).json()
    get = client.get(f"/forecast/{code.lower()}", params={"horizon": 3}).json()
    assert post["stock_code"] == code and get["stock_code"] == code
    assert post["forecast_units"] == get["forecast_units"]


@pytest.mark.parametrize("payload,field", [
    ({"stock_code": "85123A", "horizon": 0}, "horizon"),
    ({"stock_code": "85123A", "horizon": 5}, "horizon"),
    ({"stock_code": "85123A", "horizon": "two"}, "horizon"),
    ({"horizon": 1}, "stock_code"),
    ({"stock_code": "", "horizon": 1}, "stock_code"),
])
def test_invalid_input_returns_422(client, payload, field):
    r = client.post("/forecast", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "error"
    assert any(p["field"] == field for p in body["problems"])


def test_unknown_product_returns_404(client):
    r = client.post("/forecast", json={"stock_code": "NOT-A-PRODUCT", "horizon": 1})
    assert r.status_code == 404
    body = r.json()
    assert "not in the forecast scope" in body["error"]
    assert "/products" in body["hint"]


def test_unsupported_store_returns_422(client):
    r = client.post("/forecast", json={"stock_code": "85123A", "horizon": 1, "store": "London"})
    assert r.status_code == 422
    assert "no store" in r.json()["hint"]


def test_replenishment_matches_module(client):
    code = EXAMPLE_CODES[1]
    r = client.post("/replenishment", json={"stock_code": code, "on_hand": 100, "on_order": 50})
    assert r.status_code == 200, r.text
    body = r.json()
    bt = ad.load_backtest()
    stock = pd.DataFrame({"StockCode": [code], "on_hand": [100], "on_order": [50]})
    plan = ad.build_plan(bt, FUTURE[FUTURE["StockCode"] == code], 2, 1, 0.95, stock=stock)
    p = plan.iloc[0]
    assert body["reorder_point"] == pytest.approx(p["reorder_point"])
    assert body["suggested_order_qty"] == pytest.approx(p["suggested_order_qty"])
    assert body["inventory_position"] == 150


def test_replenishment_logic(client):
    code = EXAMPLE_CODES[0]
    big = client.post("/replenishment", json={"stock_code": code, "on_hand": 1e7}).json()
    assert big["reorder_now"] is False and big["suggested_order_qty"] == 0
    packed = client.post("/replenishment", json={"stock_code": code, "on_hand": 0, "pack_size": 12}).json()
    assert packed["reorder_now"] is True
    assert packed["suggested_order_qty"] % 12 == 0
    assert packed["suggested_order_qty"] >= packed["order_up_to_level"]


@pytest.mark.parametrize("payload", [
    {"stock_code": "85123A", "on_hand": -5},
    {"stock_code": "85123A", "on_hand": 10, "lead_time_weeks": 4},
    {"stock_code": "85123A", "on_hand": 10, "service_level": 1.5},
    {"stock_code": "85123A"},
])
def test_replenishment_invalid_input(client, payload):
    r = client.post("/replenishment", json=payload)
    assert r.status_code == 422
    assert r.json()["problems"]
