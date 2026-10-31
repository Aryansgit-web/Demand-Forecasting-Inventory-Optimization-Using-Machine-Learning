"""
Demand Forecasting & Inventory Decision-Support Dashboard

Run from the project root:
    streamlit run app/dashboard.py

Pages
  1. Demand Overview        actual vs forecast demand, forecast error, filters
  2. Product Forecasts      history, backtest and future forecast per product
  3. Inventory Risk         reorder point, safety stock, suggested order (stock = user input)
  4. Model Performance      MAE / RMSE / WAPE / bias vs baselines, error by product
  5. Assumptions            documented assumptions and limitations
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import app_data as ad                       # noqa: E402
from src.config import LEAD_TIME_WEEKS, REVIEW_PERIOD_WEEKS, SERVICE_LEVEL, MODEL_HORIZONS  # noqa: E402
from src.evaluation import metrics                   # noqa: E402
from src.replenishment import service_level_backtest, z_value  # noqa: E402
from src.splits import VALID_START, TEST_START       # noqa: E402

st.set_page_config(page_title="Demand Forecasting & Inventory", page_icon="📦", layout="wide")

COLORS = {"actual": "#1f2937", "model": "#2563eb", "baseline": "#f59e0b", "future": "#16a34a"}
METHOD_LABELS = {"ensemble": "Ensemble (selected)", "xgboost": "XGBoost", "random_forest": "Random Forest",
                 "ses": "Exp. smoothing", "moving_avg_4": "4-week moving avg", "naive": "Naive (last week)",
                 "seasonal_naive_4": "Seasonal naive (4 wk)", "seasonal_naive_52": "Seasonal naive (52 wk)"}
DISCLAIMER = ("Forecasts are model estimates, not ground truth. Inventory figures rest on the stated "
              "assumptions (see *Assumptions*). Use them to support, not replace, planner judgement.")


# ------------------------------------------------------------------ data (cached)
@st.cache_data(show_spinner="Loading forecasts…")
def load_all():
    backtest = ad.load_backtest()
    future = ad.load_future()
    names = ad.product_names(future)
    history = ad.load_history(names["StockCode"])
    flags = ad.decline_flags(history)
    volume = history.groupby("StockCode")["units_sold"].sum().rename("total_units").reset_index()
    names = names.merge(volume, on="StockCode").merge(flags, on="StockCode")
    names = names.sort_values("total_units", ascending=False).reset_index(drop=True)
    names["label"] = names["StockCode"] + " – " + names["Description"].fillna("")
    return backtest, future, history, names


@st.cache_data
def load_reports():
    return {
        "comparison": ad.load_report("model_comparison.csv"),
        "comparison_total": ad.load_report("model_comparison_total.csv"),
        "by_product": ad.load_report("error_by_product.csv", dtype={"StockCode": str}),
        "by_class": ad.load_report("error_by_demand_class.csv"),
        "selected": ad.load_report("selected_models.csv"),
        "card": ad.load_model_card(),
    }


@st.cache_data(show_spinner="Calculating replenishment plan…")
def cached_plan(lead_time, service_level, pack_size, moq, stock_csv: str | None):
    backtest, future, _, names = load_all()
    stock = pd.read_json(io.StringIO(stock_csv), dtype={"StockCode": str}) if stock_csv else None
    plan = ad.build_plan(backtest, future, lead_time, REVIEW_PERIOD_WEEKS, service_level,
                         stock=stock, pack_size=pack_size, min_order_qty=moq)
    plan = names[["StockCode", "Description", "declining_flag", "trend_4_vs_12"]].merge(plan, on="StockCode")
    test = backtest[backtest["split"] == "test"]
    bt = service_level_backtest(test, ad.variability(backtest, lead_time), lead_time, service_level)
    return plan, float(bt["covered"].mean())


backtest, future, history, names = load_all()
reports = load_reports()
SELECTED = reports["card"]["selected_method"]
LAST_WEEK = pd.Timestamp(reports["card"]["last_observed_week"])


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return 100 * np.abs(f - a).sum() / a.sum() if a.sum() else np.nan


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return 100 * (f - a).sum() / a.sum() if a.sum() else np.nan


def add_period_lines(fig):
    for x, label in ((VALID_START, "validation"), (TEST_START, "test"), (LAST_WEEK + pd.Timedelta(weeks=1), "future")):
        fig.add_vline(x=x, line_dash="dot", line_color="grey")
        fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, xanchor="left",
                           font=dict(size=11, color="grey"))
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📦 Demand & Inventory")
page = st.sidebar.radio("Page", ["Demand Overview", "Product Forecasts", "Inventory Risk",
                                 "Model Performance", "Assumptions"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Selected model: **{METHOD_LABELS.get(SELECTED, SELECTED)}**  \n"
                   f"Products: **{len(names):,}**  \nLast observed week: **{LAST_WEEK.date()}**")
st.sidebar.caption("Data: UCI Online Retail (UK gift retailer, Dec 2010 – Dec 2011).")


# ================================================================== 1. Demand overview
if page == "Demand Overview":
    st.title("Demand Overview")
    st.caption("Total weekly demand for the forecast products: actual vs forecast, and forecast error.")

    c1, c2, c3 = st.columns(3)
    h = c1.selectbox("Forecast horizon (weeks ahead)", MODEL_HORIZONS, index=0)
    periods = c2.multiselect("Evaluation period", ["validation", "test"], default=["validation", "test"])
    baseline = c3.selectbox("Compare with baseline", ["moving_avg_4", "naive"],
                            format_func=lambda m: METHOD_LABELS[m])
    chosen = st.multiselect("Products (leave empty for all)", names["label"], default=[],
                            placeholder="All forecast products")
    codes = names.loc[names["label"].isin(chosen), "StockCode"] if chosen else names["StockCode"]

    bt = backtest[(backtest["horizon"] == h) & backtest["split"].isin(periods) & backtest["StockCode"].isin(codes)]
    if bt.empty:
        st.warning("No rows for this selection. Pick at least one evaluation period.")
        st.stop()

    k = st.columns(4)
    k[0].metric("Actual units", f"{bt['units_sold'].sum():,.0f}")
    k[1].metric("Forecast units", f"{bt['forecast'].sum():,.0f}",
                delta=f"{bias(bt['units_sold'], bt['forecast']):+.1f}% bias", delta_color="off")
    k[2].metric("Product-level WAPE", f"{wape(bt['units_sold'], bt['forecast']):.1f}%",
                delta=f"{wape(bt['units_sold'], bt['forecast']) - wape(bt['units_sold'], bt[baseline]):+.1f} pts vs baseline",
                delta_color="inverse")
    tot = bt.groupby("week_start")[["units_sold", "forecast", baseline]].sum().reset_index()
    k[3].metric("Total-demand WAPE", f"{wape(tot['units_sold'], tot['forecast']):.1f}%",
                delta=f"{wape(tot['units_sold'], tot['forecast']) - wape(tot['units_sold'], tot[baseline]):+.1f} pts vs baseline",
                delta_color="inverse")

    hist = history[history["StockCode"].isin(codes)].groupby("week_start")["units_sold"].sum().reset_index()
    fut = future[(future["StockCode"].isin(codes))].groupby("week_start")["forecast"].sum().reset_index()

    fig = go.Figure()
    fig.add_scatter(x=hist["week_start"], y=hist["units_sold"], name="Actual", mode="lines+markers",
                    line=dict(color=COLORS["actual"], width=2))
    fig.add_scatter(x=tot["week_start"], y=tot["forecast"], name=f"{METHOD_LABELS[SELECTED]} (h={h})",
                    mode="lines+markers", line=dict(color=COLORS["model"]))
    fig.add_scatter(x=tot["week_start"], y=tot[baseline], name=METHOD_LABELS[baseline], mode="lines",
                    line=dict(color=COLORS["baseline"], dash="dash"))
    fig.add_scatter(x=fut["week_start"], y=fut["forecast"], name="Future forecast (h=1–4)",
                    mode="lines+markers", line=dict(color=COLORS["future"], dash="dot"))
    add_period_lines(fig)
    fig.update_layout(height=430, yaxis_title="Units per week", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    err = (bt.groupby("week_start")
             .apply(lambda g: pd.Series({"Model": wape(g["units_sold"], g["forecast"]),
                                         "Baseline": wape(g["units_sold"], g[baseline])}),
                    include_groups=False).reset_index()
             .melt(id_vars="week_start", var_name="method", value_name="WAPE %"))
    fig2 = px.bar(err, x="week_start", y="WAPE %", color="method", barmode="group",
                  color_discrete_map={"Model": COLORS["model"], "Baseline": COLORS["baseline"]},
                  title="Product-level forecast error (WAPE) by week")
    fig2.update_layout(height=330, xaxis_title=None, legend_title=None, margin=dict(t=50))
    st.plotly_chart(fig2, width="stretch")

    st.info("**How to read this:** WAPE = total absolute error ÷ total actual units. Product-level WAPE is high "
            "because weekly demand per product is noisy; summed across products the forecast is much more "
            "accurate (total-demand WAPE). Bias above 0 means over-forecasting.")
    st.caption(DISCLAIMER)


# ================================================================== 2. Product forecasts
elif page == "Product Forecasts":
    st.title("Product Forecasts")
    c1, c2 = st.columns([3, 1])
    label = c1.selectbox("Product (sorted by sales volume; type to search)", names["label"])
    h = c2.selectbox("Backtest horizon", MODEL_HORIZONS, index=0)
    row = names[names["label"] == label].iloc[0]
    code = row["StockCode"]

    ph = history[history["StockCode"] == code]
    pb = backtest[(backtest["StockCode"] == code) & (backtest["horizon"] == h)]
    pf = future[future["StockCode"] == code].sort_values("horizon")
    pe = reports["by_product"][reports["by_product"]["StockCode"] == code]

    if row["declining_flag"]:
        st.warning(f"⚠️ Demand is falling sharply: the last 4 weeks average {row['mean_last_4w']:.0f} units vs "
                   f"{row['mean_last_12w']:.0f} over 12 weeks ({row['trend_4_vs_12']:.0%}). The product may be "
                   "phasing out — forecasts lag such drops. Review before ordering.")

    k = st.columns(4)
    k[0].metric("Avg weekly units (last 4 wk)", f"{row['mean_last_4w']:,.0f}")
    k[1].metric("Forecast next week", f"{pf.loc[pf['horizon'] == 1, 'forecast'].sum():,.0f}")
    k[2].metric("Forecast next 4 weeks", f"{pf['forecast'].sum():,.0f}")
    if not pe.empty:
        k[3].metric("Test WAPE (h=1)", f"{pe['WAPE_%'].iat[0]:.0f}%",
                    delta=f"{pe['WAPE_%'].iat[0] - pe['ma4_WAPE_%'].iat[0]:+.0f} pts vs moving avg",
                    delta_color="inverse")

    fig = go.Figure()
    fig.add_scatter(x=ph["week_start"], y=ph["units_sold"], name="Actual", mode="lines+markers",
                    line=dict(color=COLORS["actual"]))
    fig.add_scatter(x=pb["week_start"], y=pb["forecast"], name=f"{METHOD_LABELS[SELECTED]} (h={h})",
                    mode="lines+markers", line=dict(color=COLORS["model"]))
    fig.add_scatter(x=pb["week_start"], y=pb["moving_avg_4"], name=METHOD_LABELS["moving_avg_4"],
                    mode="lines", line=dict(color=COLORS["baseline"], dash="dash"))
    fig.add_scatter(x=pf["week_start"], y=pf["forecast"], name="Future forecast", mode="lines+markers",
                    line=dict(color=COLORS["future"], dash="dot"), marker=dict(size=9))
    add_period_lines(fig)
    fig.update_layout(height=450, yaxis_title="Units per week", hovermode="x unified",
                      title=f"{code} – {row['Description']}", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Future forecast by method")
    tbl = pf[["week_start", "horizon", "forecast", "xgboost", "random_forest", "ses"]].copy()
    tbl["week_start"] = tbl["week_start"].dt.date
    tbl.columns = ["Week starting", "Weeks ahead", "Selected (ensemble)", "XGBoost", "Random Forest", "Exp. smoothing"]
    st.dataframe(tbl.round(1), hide_index=True, width="stretch")
    if (pf["week_start"] == pd.Timestamp("2011-12-26")).any():
        st.caption("The week of 26 Dec 2011 includes two bank holidays; in 2010 the shop was closed that week.")

    with st.expander("Backtest detail (validation and test weeks)"):
        d = pb[["week_start", "split", "units_sold", "forecast", "moving_avg_4", "naive"]].copy()
        d["week_start"] = d["week_start"].dt.date
        st.dataframe(d.round(1), hide_index=True, width="stretch")
    st.caption(DISCLAIMER)


# ================================================================== 3. Inventory risk
elif page == "Inventory Risk":
    st.title("Inventory Risk & Replenishment")
    st.caption("Periodic review (weekly), order-up-to policy. Current stock is **not in the dataset** — "
               "enter or upload it to get order quantities and stockout risk.")

    with st.expander("⚙️ Assumptions for this calculation", expanded=True):
        c = st.columns(4)
        lead_time = c[0].selectbox("Supplier lead time (weeks)", [1, 2, 3], index=[1, 2, 3].index(LEAD_TIME_WEEKS))
        service_level = c[1].slider("Target service level", 0.80, 0.99, SERVICE_LEVEL, 0.01)
        pack_size = c[2].number_input("Pack size (units)", 1, 1000, 1)
        moq = c[3].number_input("Minimum order qty", 0, 10000, 0)

    st.subheader("Current stock")
    mode = st.radio("Stock input", ["None yet — show reorder points only", "Upload CSV", "Type in a few products"],
                    horizontal=True, key="stock_mode")
    stock = None
    template = pd.DataFrame({"StockCode": names["StockCode"].head(5), "on_hand": [None] * 5, "on_order": [0] * 5})
    if mode == "Upload CSV":
        st.download_button("Download CSV template (all products)",
                           pd.DataFrame({"StockCode": names["StockCode"], "on_hand": "", "on_order": 0})
                           .to_csv(index=False), "stock_template.csv", "text/csv")
        up = st.file_uploader("CSV with columns StockCode, on_hand, on_order (optional)", type="csv")
        if up is not None:
            s = pd.read_csv(up, dtype={"StockCode": str})
            missing = {"StockCode", "on_hand"} - set(s.columns)
            if missing:
                st.error(f"Missing column(s): {', '.join(sorted(missing))}")
            else:
                s["on_hand"] = pd.to_numeric(s["on_hand"], errors="coerce")
                bad = s["on_hand"].isna() | (s["on_hand"] < 0)
                if bad.any():
                    st.warning(f"{bad.sum()} row(s) ignored: on_hand must be a number ≥ 0.")
                unknown = ~s["StockCode"].isin(names["StockCode"])
                if unknown.any():
                    st.warning(f"{unknown.sum()} unknown StockCode(s) ignored.")
                stock = s[~bad & ~unknown]
    elif mode == "Type in a few products":
        picks = st.multiselect("Products", names["label"], default=names["label"].head(5).tolist())
        base = pd.DataFrame({"StockCode": names.set_index("label").loc[picks, "StockCode"].values,
                             "on_hand": 0, "on_order": 0})
        edited = st.data_editor(base, hide_index=True, width="stretch",
                                column_config={"StockCode": st.column_config.TextColumn(disabled=True),
                                               "on_hand": st.column_config.NumberColumn(min_value=0, step=1),
                                               "on_order": st.column_config.NumberColumn(min_value=0, step=1)})
        stock = edited if len(edited) else None

    stock_json = stock[["StockCode", "on_hand"] + (["on_order"] if "on_order" in stock else [])].to_json() \
        if stock is not None and len(stock) else None
    plan, achieved = cached_plan(lead_time, service_level, int(pack_size), int(moq), stock_json)

    st.info(f"**Check of the safety-stock assumption:** in the unseen test period, a {service_level:.0%} target with a "
            f"{lead_time}-week lead time covered **{achieved:.1%}** of lead-time demand cycles. "
            "If this is below target, choose a higher service level.")

    if stock_json is None:
        k = st.columns(3)
        k[0].metric("Products", f"{len(plan):,}")
        k[1].metric("Total safety stock (units)", f"{plan['safety_stock'].sum():,.0f}")
        k[2].metric("Declining products to review", f"{int(plan['declining_flag'].sum())}")
        show = plan[["StockCode", "Description", "forecast_next_week", "lead_time_demand", "safety_stock",
                     "reorder_point", "order_up_to_level", "declining_flag"]]
    else:
        have = plan[plan["inventory_position"].notna()].copy()
        k = st.columns(4)
        k[0].metric("Products with stock entered", f"{len(have):,}")
        k[1].metric("Reorder now", f"{int(have['reorder_now'].fillna(False).astype(bool).sum())}")
        k[2].metric("High stockout risk (>20%)", f"{int((have['stockout_risk_lead_time'] > 0.2).sum())}")
        k[3].metric("Suggested order (units)", f"{have['suggested_order_qty'].sum():,.0f}")
        have = have.sort_values("stockout_risk_lead_time", ascending=False)
        top = have.head(15)
        fig = px.bar(top, x="stockout_risk_lead_time", y="StockCode", orientation="h",
                     color="stockout_risk_lead_time", color_continuous_scale="Reds", range_color=(0, 1),
                     hover_data=["Description", "inventory_position", "reorder_point"],
                     title="Stockout risk during the lead time (highest first)")
        fig.update_layout(height=420, yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                          xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        show = have[["StockCode", "Description", "on_hand", "on_order", "inventory_position", "lead_time_demand",
                     "reorder_point", "order_up_to_level", "reorder_now", "suggested_order_qty",
                     "stockout_risk_lead_time", "weeks_of_cover", "declining_flag"]]

    st.dataframe(show.round(2), hide_index=True, width="stretch",
                 column_config={"declining_flag": st.column_config.CheckboxColumn("Declining?"),
                                "stockout_risk_lead_time": st.column_config.ProgressColumn(
                                    "Stockout risk", min_value=0, max_value=1, format="percent")}
                 if "stockout_risk_lead_time" in show else
                 {"declining_flag": st.column_config.CheckboxColumn("Declining?")})
    st.download_button("Download plan (CSV)", show.to_csv(index=False), "replenishment_plan.csv", "text/csv")

    with st.expander("How the numbers are calculated"):
        st.markdown(f"""
| Quantity | Formula |
|---|---|
| Lead-time demand | sum of forecasts for the next {lead_time} week(s) |
| Safety stock | z × σ over lead time + review week; z = {z_value(service_level):.2f} |
| Reorder point | lead-time demand + z × σ(lead time) |
| Order-up-to level | forecast over lead time + 1 review week + safety stock |
| Suggested order | max(0, order-up-to − (on hand + on order)), rounded up to pack size, MOQ applied |
| Stockout risk | P(lead-time demand > inventory position), normal approximation |

σ = RMSE of out-of-sample cumulative forecast errors per product, measured on the validation period.
Products flagged **Declining** have recent demand below 50% of their 12-week average; their forecasts may be too high.
""")
    st.caption(DISCLAIMER)


# ================================================================== 4. Model performance
elif page == "Model Performance":
    st.title("Model Performance")
    card = reports["card"]
    st.caption(f"Selected method: **{METHOD_LABELS.get(card['selected_method'])}**. Selection rule: {card['selection_rule']}. "
               "The test period was used once, for the final comparison.")

    comp = reports["comparison"]
    c1, c2, c3 = st.columns(3)
    h = c1.selectbox("Horizon", MODEL_HORIZONS, index=0)
    split = c2.selectbox("Period", ["test", "validation"])
    level = c3.radio("Level", ["Product-week", "Total demand"], horizontal=True)
    src = comp if level == "Product-week" else reports["comparison_total"]
    t = src[(src["horizon"] == h) & (src["split"] == split) & src["WAPE_%"].notna()].copy()
    t["Method"] = t["method"].map(METHOD_LABELS)
    t = t.sort_values("WAPE_%")

    cols = ["Method", "MAE", "RMSE", "WAPE_%", "Bias_%"] + (["WAPE_improvement_vs_ma4_%"] if "WAPE_improvement_vs_ma4_%" in t else [])
    st.dataframe(t[cols].round(2), hide_index=True, width="stretch")
    st.caption("Seasonal naive (52 weeks) is excluded: the data has no previous year, so it has 0% coverage.")

    c1, c2 = st.columns(2)
    bar_colors = np.where(t["method"].eq(card["selected_method"]), COLORS["model"], "#9ca3af")
    for col, metric, title in ((c1, "WAPE_%", "WAPE (%) — lower is better"),
                               (c2, "Bias_%", "Bias (%) — closer to 0 is better")):
        fig = go.Figure(go.Bar(x=t[metric], y=t["Method"], orientation="h", marker_color=bar_colors,
                               text=t[metric].round(1), textposition="outside", cliponaxis=False))
        fig.add_vline(x=0, line_color="black")
        fig.update_layout(title=title, height=360, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40), xaxis=dict(automargin=True))
        col.plotly_chart(fig, width="stretch")
    st.caption("Blue bar = selected model. Both charts list methods in the same order (best WAPE first).")

    st.subheader("Error by product (test, h = 1)")
    bp = reports["by_product"].copy()
    st.write(f"The selected model beats the 4-week moving average on **{bp['beats_ma4'].mean():.0%}** of products; "
             "its advantage is concentrated in higher-volume products and in lower bias.")
    lim = bp.assign(model=bp["WAPE_%"].clip(upper=200), baseline=bp["ma4_WAPE_%"].clip(upper=200))
    fig = px.scatter(lim, x="baseline", y="model", hover_data=["StockCode", "Description", "MAE"], opacity=0.5,
                     labels={"baseline": "4-week moving avg WAPE % (clipped 200)", "model": "Selected model WAPE %"},
                     title="Each dot is a product — below the line means the model is better")
    fig.add_shape(type="line", x0=0, y0=0, x1=200, y1=200, line=dict(dash="dash", color="black"))
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bp[["StockCode", "Description", "MAE", "WAPE_%", "Bias_%", "ma4_WAPE_%"]].head(20).round(1),
                 hide_index=True, width="stretch")

    st.subheader("Error by demand pattern (test, h = 1)")
    st.dataframe(reports["by_class"].round(1), hide_index=True, width="stretch")
    st.caption("Lumpy / intermittent products (few in scope) include items being phased out, where forecasts lag the drop.")

    with st.expander("Chosen configuration per model family (validation)"):
        st.dataframe(reports["selected"][["horizon", "model", "target_mode", "params", "WAPE_%", "Bias_%"]].round(2),
                     hide_index=True, width="stretch")
    st.caption(DISCLAIMER)


# ================================================================== 5. Assumptions
else:
    st.title("Assumptions & Limitations")
    path = ad.REPORTS / "assumptions.md"
    st.markdown(path.read_text(encoding="utf-8") if path.exists() else "Run notebook 03 to create reports/assumptions.md.")
    st.markdown("""
### How to use these outputs
- Treat forecasts as **estimates with error**. Typical product-level error is 60–70% of weekly volume; totals are far more accurate.
- Enter **real stock** before acting on order quantities; nothing about stock is inferred from the data.
- Review **declining** products manually; the models are slow to react to phase-outs.
- Re-run notebooks 01–03 when new sales data arrives to refresh forecasts and safety stock.
""")
