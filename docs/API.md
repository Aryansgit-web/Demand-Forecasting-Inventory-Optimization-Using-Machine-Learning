# Forecast API: setup and usage

A FastAPI prototype that serves weekly demand forecasts and replenishment suggestions for the 978 forecast products.
It reads the outputs of notebooks 01–03 (`data/processed/forecasts_*.csv`, `models/`, `reports/`) and never retrains anything.

## Setup

```powershell
pip install -r requirements.txt          # fastapi, uvicorn, httpx, pytest included
# run notebooks 01, 02, 03 first so the forecast files exist
uvicorn api.main:app --reload            # from the project root
```

- **API:** http://127.0.0.1:8000
- **Interactive docs (try every endpoint in the browser):** http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status, selected model, horizons, forecast weeks |
| GET | `/products?search=&limit=` | Find valid StockCodes (search code or description) |
| POST | `/forecast` | Forecast for a product, store and horizon |
| GET | `/forecast/{stock_code}?horizon=&store=` | Same as POST, for quick browser use |
| POST | `/replenishment` | Reorder point, safety stock and suggested order for one product, given current stock |

### `POST /forecast`

| Field | Type | Rules | Default |
|---|---|---|---|
| `stock_code` | string | Required. Case-insensitive. Must be one of the 978 forecast products. | — |
| `horizon` | integer | 1–4 weeks ahead | 1 |
| `store` | string | Only `"ALL"`. The dataset has no store field. | `"ALL"` |

Request:

```bash
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" \
     -d '{"stock_code": "85123A", "horizon": 2}'
```

Response (200):

```json
{
  "status": "ok",
  "stock_code": "85123A",
  "description": "WHITE HANGING HEART T-LIGHT HOLDER",
  "store": "ALL",
  "method": "ensemble",
  "origin_week": "2011-11-28",
  "target_week": "2011-12-12",
  "horizon": 2,
  "forecast_units": 871.64,
  "interval_90": {"lower": 465.44, "upper": 1277.84},
  "path": [
    {"week_start": "2011-12-05", "horizon": 1, "forecast_units": 842.77},
    {"week_start": "2011-12-12", "horizon": 2, "forecast_units": 871.64}
  ],
  "components": {"xgboost": 905.29, "random_forest": 989.66, "ses": 719.97},
  "model_validation_wape_pct": 64.46,
  "model_test_wape_pct": 61.69,
  "warnings": [],
  "note": "Model estimate, not ground truth. Interval is approximate ..."
}
```

| Field | Meaning |
|---|---|
| `forecast_units` | Expected units in `target_week`, forecast from `origin_week`, the last observed week |
| `path` | Forecasts for every week up to the horizon. Sum them for lead-time demand. |
| `interval_90` | Approximate 90% range: forecast ± 1.645 × the product's validation RMSE at that horizon, floored at 0 |
| `components` | The three models averaged into the selected ensemble |
| `model_*_wape_pct` | Typical product-level error of the method at this horizon, as a % of volume |
| `warnings` | Plain-English cautions, such as sharply declining demand or the Christmas-week closure |

### `POST /replenishment`

| Field | Rules | Default |
|---|---|---|
| `stock_code` | Required, as above | — |
| `on_hand` | Required, ≥ 0. Current stock is **not in the dataset**, so you must supply it. | — |
| `on_order`, `backorders` | ≥ 0 | 0 |
| `lead_time_weeks` | 1–3 (lead time + 1 review week ≤ 4 forecast weeks) | 2 |
| `service_level` | 0.5–0.999 | 0.95 |
| `pack_size` | ≥ 1; orders round up to a multiple | 1 |
| `min_order_qty` | ≥ 0 | 0 |

```bash
curl -X POST http://127.0.0.1:8000/replenishment -H "Content-Type: application/json" \
     -d '{"stock_code": "85123A", "on_hand": 800, "on_order": 200, "pack_size": 12}'
```

```json
{
  "status": "ok",
  "stock_code": "85123A",
  "assumptions": {"lead_time_weeks": 2, "review_period_weeks": 1, "service_level": 0.95, "pack_size": 12, "min_order_qty": 0},
  "inputs": {"on_hand": 800.0, "on_order": 200.0, "backorders": 0},
  "lead_time_demand": 1714.41,
  "safety_stock": 514.0,
  "reorder_point": 2251.0,
  "order_up_to_level": 3057.0,
  "inventory_position": 1000.0,
  "reorder_now": true,
  "suggested_order_qty": 2064.0,
  "stockout_risk_lead_time": 0.99,
  "weeks_of_cover": 1.19,
  "warnings": [],
  "note": "..."
}
```

## Error responses

Every error has `"status": "error"`, a readable `error` message and, where useful, a `hint`.

| Situation | HTTP | Example body |
|---|---|---|
| Field missing, wrong type or out of range | 422 | `{"status":"error","error":"Invalid request","problems":[{"field":"horizon","message":"Input should be less than or equal to 4","received":7}],"hint":"See /docs ..."}` |
| Unknown product | 404 | `{"status":"error","error":"Product 'ABC123' is not in the forecast scope.","hint":"Use GET /products?search=... to find valid StockCodes. ..."}` |
| Store other than ALL | 422 | `{"status":"error","error":"Store 'LONDON' is not supported.","hint":"The Online Retail dataset has no store/location field; ...","supported_stores":["ALL"]}` |

## Python example

```python
import requests
r = requests.post("http://127.0.0.1:8000/forecast", json={"stock_code": "22197", "horizon": 4})
if r.ok:
    f = r.json()
    print(f["target_week"], f["forecast_units"], f["interval_90"], f["warnings"])
else:
    print(r.status_code, r.json()["error"], r.json().get("hint"))
```

## Tests

```powershell
python -m pytest -q tests
```

The suite has 71 tests and takes about 10 seconds. The tests skip automatically if notebooks 01–03 haven't been run.

| File | What it checks |
|---|---|
| `tests/test_api.py` | API forecasts equal the stored `forecasts_future.csv` for 10 products × 4 horizons; interval and path are consistent; GET and POST agree; product codes are case-insensitive; replenishment matches `src/replenishment.py`; pack-size rounding and the reorder decision; every invalid input returns 404 or 422 with a clear message |
| `tests/test_stored_examples.py` | Reloading the saved models in `models/` reproduces the stored forecasts; the ensemble equals the mean of its parts; stored metrics recompute exactly from `forecasts_backtest.csv`; the model was chosen on **validation** (not test) and beats the moving-average baseline there |

## Limitations

- **No live data:** forecasts are fixed at the last observed week (28 Nov 2011). Re-run notebooks 01–03 with new data to refresh them.
- **No store dimension:** the data has no store field, so `store` exists for interface compatibility only.
- **Approximate interval:** `interval_90` assumes normally distributed errors and is less reliable for intermittent products.
- **Stock and lead time:** these are user inputs and assumptions, never inferred from the data.
- **Prototype security:** there's no authentication or rate limiting. Don't expose the API publicly as it is.
