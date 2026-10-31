"""
Demand Forecasting API (FastAPI prototype)

Run from the project root:
    uvicorn api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs

Endpoints
  GET  /health                     service status and model metadata
  GET  /products                   list / search forecastable products
  POST /forecast                   forecast for a product (+ store) and horizon
  GET  /forecast/{stock_code}      same, as a GET with query parameters
  POST /replenishment              reorder point and suggested order for one product

Forecasts are served from the outputs of notebook 03 (production models trained on
all observed weeks, forecast origin = last observed week). Nothing is retrained here.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import app_data as ad                                   # noqa: E402
from src.config import (LEAD_TIME_WEEKS, MODEL_HORIZONS,          # noqa: E402
                        REVIEW_PERIOD_WEEKS, SERVICE_LEVEL)
from src.replenishment import z_value                            # noqa: E402

MAX_H = max(MODEL_HORIZONS)
SUPPORTED_STORES = ("ALL",)
INTERVAL_LEVEL = 0.90
NOTE = ("Model estimate, not ground truth. Interval is approximate (normal, from out-of-sample "
        "validation errors). Inventory outputs depend on the stated assumptions.")

STATE: dict = {}


def load_state() -> dict:
    backtest = ad.load_backtest()
    future = ad.load_future()
    names = ad.product_names(future)
    history = ad.load_history(names["StockCode"])
    flags = ad.decline_flags(history).set_index("StockCode")

    val = backtest[backtest["split"] == "validation"]
    sigma_h = (val.assign(e2=(val["forecast"] - val["units_sold"]) ** 2)
                  .groupby(["StockCode", "horizon"])["e2"].mean().pow(0.5))
    comp = ad.load_report("model_comparison.csv")
    card = ad.load_model_card()
    sel = card["selected_method"]
    val_wape = (comp[(comp["split"] == "validation") & (comp["method"] == sel)]
                .set_index("horizon")["WAPE_%"].to_dict())
    test_wape = (comp[(comp["split"] == "test") & (comp["method"] == sel)]
                 .set_index("horizon")["WAPE_%"].to_dict())
    return {"backtest": backtest, "future": future, "names": names.set_index("StockCode"),
            "flags": flags, "sigma_h": sigma_h, "card": card,
            "val_wape": val_wape, "test_wape": test_wape}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_state())
    yield
    STATE.clear()


app = FastAPI(title="Demand Forecasting & Replenishment API", version="1.0.0", lifespan=lifespan,
              description="Weekly product demand forecasts (UCI Online Retail) and replenishment "
                          "suggestions. See docs/API.md for usage and assumptions.")


# ------------------------------------------------------------------ errors
def error(status: int, message: str, hint: str | None = None, **extra):
    body = {"error": message}
    if hint:
        body["hint"] = hint
    body.update(extra)
    raise HTTPException(status_code=status, detail=body)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"status": "error", **detail})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = []
    for e in exc.errors():
        field = ".".join(str(p) for p in e["loc"] if p not in ("body", "query", "path"))
        problems.append({"field": field or "request", "message": e["msg"],
                         "received": e.get("input") if not isinstance(e.get("input"), dict) else None})
    return JSONResponse(status_code=422, content={
        "status": "error", "error": "Invalid request", "problems": problems,
        "hint": "See /docs for the expected fields and allowed values."})


# ------------------------------------------------------------------ schemas
class ForecastRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20, examples=["85123A"],
                            description="Product StockCode (case-insensitive).")
    horizon: int = Field(1, ge=1, le=MAX_H, description=f"Weeks ahead, 1–{MAX_H}.")
    store: str = Field("ALL", description="Store / location. The dataset has no store field, so only 'ALL'.")

    @field_validator("stock_code")
    @classmethod
    def clean_code(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("store")
    @classmethod
    def clean_store(cls, v: str) -> str:
        return v.strip().upper()


class ReplenishmentRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=20, examples=["85123A"])
    on_hand: float = Field(..., ge=0, description="Units currently in stock (required; not in the dataset).")
    on_order: float = Field(0, ge=0, description="Units already ordered and not yet received.")
    backorders: float = Field(0, ge=0)
    lead_time_weeks: int = Field(LEAD_TIME_WEEKS, ge=1, le=MAX_H - REVIEW_PERIOD_WEEKS,
                                 description="Supplier lead time (assumption).")
    service_level: float = Field(SERVICE_LEVEL, ge=0.5, le=0.999, description="Target cycle service level.")
    pack_size: int = Field(1, ge=1)
    min_order_qty: int = Field(0, ge=0)
    store: str = Field("ALL")

    @field_validator("stock_code", "store")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip().upper()


class WeekForecast(BaseModel):
    week_start: str
    horizon: int
    forecast_units: float


class ForecastResponse(BaseModel):
    status: Literal["ok"] = "ok"
    stock_code: str
    description: str | None
    store: str
    method: str
    origin_week: str
    target_week: str
    horizon: int
    forecast_units: float
    interval_90: dict
    path: list[WeekForecast]
    components: dict
    model_validation_wape_pct: float | None
    model_test_wape_pct: float | None
    warnings: list[str]
    note: str


# ------------------------------------------------------------------ helpers
def check_store(store: str):
    if store not in SUPPORTED_STORES:
        error(422, f"Store '{store}' is not supported.",
              hint="The Online Retail dataset has no store/location field; forecasts are product-level. "
                   "Use store='ALL' or omit it.", supported_stores=list(SUPPORTED_STORES))


def check_product(code: str):
    if code not in STATE["names"].index:
        error(404, f"Product '{code}' is not in the forecast scope.",
              hint="Use GET /products?search=... to find valid StockCodes. Only products sold in ≥80% of "
                   "training weeks are forecast.")


def product_warnings(code: str, target_week: pd.Timestamp) -> list[str]:
    w = []
    f = STATE["flags"].loc[code]
    if bool(f["declining_flag"]):
        w.append(f"Demand is falling sharply (last 4 weeks = {f['trend_4_vs_12']:.0%} of the 12-week average); "
                 "the forecast may be too high. Review before ordering.")
    if target_week == pd.Timestamp("2011-12-26"):
        w.append("Week of 26 Dec 2011 includes two bank holidays; the shop was closed that week in 2010.")
    return w


def build_forecast(code: str, horizon: int, store: str) -> ForecastResponse:
    check_store(store)
    check_product(code)
    fut = STATE["future"]
    rows = fut[(fut["StockCode"] == code) & (fut["horizon"] <= horizon)].sort_values("horizon")
    target = rows[rows["horizon"] == horizon].iloc[0]
    fc = float(target["forecast"])
    sigma = STATE["sigma_h"].get((code, horizon), np.nan)
    z = z_value(0.5 + INTERVAL_LEVEL / 2)
    lower, upper = (max(0.0, fc - z * sigma), fc + z * sigma) if np.isfinite(sigma) else (None, None)
    return ForecastResponse(
        stock_code=code, description=STATE["names"].loc[code, "Description"], store=store,
        method=STATE["card"]["selected_method"],
        origin_week=str(target["origin_week"].date()), target_week=str(target["week_start"].date()),
        horizon=horizon, forecast_units=round(fc, 2),
        interval_90={"lower": None if lower is None else round(lower, 2),
                     "upper": None if upper is None else round(upper, 2)},
        path=[WeekForecast(week_start=str(r.week_start.date()), horizon=int(r.horizon),
                           forecast_units=round(float(r.forecast), 2)) for r in rows.itertuples()],
        components={m: round(float(target[m]), 2) for m in ("xgboost", "random_forest", "ses")},
        model_validation_wape_pct=round(STATE["val_wape"].get(horizon, np.nan), 2),
        model_test_wape_pct=round(STATE["test_wape"].get(horizon, np.nan), 2),
        warnings=product_warnings(code, target["week_start"]), note=NOTE)


# ------------------------------------------------------------------ endpoints
@app.get("/health", tags=["service"])
def health():
    c = STATE["card"]
    return {"status": "ok", "selected_method": c["selected_method"], "horizons": c["horizons"],
            "products": int(len(STATE["names"])), "last_observed_week": c["last_observed_week"],
            "forecast_weeks": sorted(STATE["future"]["week_start"].dt.date.astype(str).unique().tolist())}


@app.get("/products", tags=["products"])
def products(search: str | None = Query(None, max_length=50, description="Text in StockCode or description"),
             limit: int = Query(20, ge=1, le=1000)):
    n = STATE["names"].reset_index()
    if search:
        s = search.strip().upper()
        n = n[n["StockCode"].str.upper().str.contains(s, regex=False)
              | n["Description"].fillna("").str.upper().str.contains(s, regex=False)]
    return {"status": "ok", "count": int(len(n)),
            "products": n.head(limit)[["StockCode", "Description"]]
                         .rename(columns={"StockCode": "stock_code", "Description": "description"})
                         .to_dict(orient="records")}


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(req: ForecastRequest):
    return build_forecast(req.stock_code, req.horizon, req.store)


@app.get("/forecast/{stock_code}", response_model=ForecastResponse, tags=["forecast"])
def forecast_get(stock_code: str, horizon: int = Query(1, ge=1, le=MAX_H), store: str = Query("ALL")):
    return build_forecast(stock_code.strip().upper(), horizon, store.strip().upper())


@app.post("/replenishment", tags=["inventory"])
def replenishment(req: ReplenishmentRequest):
    check_store(req.store)
    check_product(req.stock_code)
    stock = pd.DataFrame({"StockCode": [req.stock_code], "on_hand": [req.on_hand],
                          "on_order": [req.on_order], "backorders": [req.backorders]})
    future = STATE["future"][STATE["future"]["StockCode"] == req.stock_code]
    plan = ad.build_plan(STATE["backtest"], future, req.lead_time_weeks, REVIEW_PERIOD_WEEKS,
                         req.service_level, stock=stock, pack_size=req.pack_size,
                         min_order_qty=req.min_order_qty)
    p = plan[plan["StockCode"] == req.stock_code].iloc[0]
    num = lambda x: None if pd.isna(x) else round(float(x), 2)          # noqa: E731
    return {
        "status": "ok", "stock_code": req.stock_code,
        "description": STATE["names"].loc[req.stock_code, "Description"],
        "assumptions": {"lead_time_weeks": req.lead_time_weeks, "review_period_weeks": REVIEW_PERIOD_WEEKS,
                        "service_level": req.service_level, "pack_size": req.pack_size,
                        "min_order_qty": req.min_order_qty},
        "inputs": {"on_hand": req.on_hand, "on_order": req.on_order, "backorders": req.backorders},
        "lead_time_demand": num(p["lead_time_demand"]),
        "safety_stock": num(p["safety_stock"]),
        "reorder_point": num(p["reorder_point"]),
        "order_up_to_level": num(p["order_up_to_level"]),
        "inventory_position": num(p["inventory_position"]),
        "reorder_now": bool(p["reorder_now"]),
        "suggested_order_qty": num(p["suggested_order_qty"]),
        "stockout_risk_lead_time": num(p["stockout_risk_lead_time"]),
        "weeks_of_cover": num(p["weeks_of_cover"]),
        "warnings": product_warnings(req.stock_code, pd.Timestamp("1900-01-01")),
        "note": NOTE,
    }
