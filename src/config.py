"""
Project-wide settings and the documented business assumptions.

The dataset has NO inventory, supplier, lead-time, pack-size or promotion
fields. Every value below is an ASSUMPTION chosen for the prototype. None is
taken from the data; change them here (or in the dashboard) when real values
are known.
"""

RANDOM_SEED = 42

# Forecast horizons modelled directly (one model per horizon).
# 1-4 weeks lets us sum forecasts over any lead time up to 4 weeks.
MODEL_HORIZONS = (1, 2, 3, 4)
HEADLINE_HORIZONS = (1, 4)          # reported in the main comparison tables

# ---------------------------------------------------------------- inventory
LEAD_TIME_WEEKS = 2        # ASSUMPTION: supplier lead time, same for all products
REVIEW_PERIOD_WEEKS = 1    # ASSUMPTION: stock is reviewed and ordered weekly
SERVICE_LEVEL = 0.95       # ASSUMPTION: target cycle service level (P(no stockout in a cycle))
PACK_SIZE = 1              # ASSUMPTION: no pack-size data; order in single units
MIN_ORDER_QTY = 0          # ASSUMPTION: no supplier minimum order quantity data
MAX_LEAD_TIME_WEEKS = max(MODEL_HORIZONS)

# Current stock on hand / on order is NOT in the data. The replenishment
# module takes it as an input; it is never estimated or invented.

ASSUMPTIONS = [
    ("Lead time", f"{LEAD_TIME_WEEKS} weeks for every product",
     "No supplier data. Configurable from 1 to 4 weeks (the model horizons)."),
    ("Review period", f"{REVIEW_PERIOD_WEEKS} week",
     "Orders are placed weekly (periodic review, order-up-to policy)."),
    ("Service level", f"{SERVICE_LEVEL:.0%} cycle service level",
     "Probability of not running out during lead time + review period."),
    ("Demand variability", "RMSE of out-of-sample cumulative forecast errors per product",
     "Measured on the validation period with models trained on the train period only."),
    ("Error distribution", "Normal approximation",
     "z = NormalDist().inv_cdf(service level). Approximate for intermittent items."),
    ("Current stock", "User input only",
     "Not in the dataset. Outputs that need it stay empty until stock is supplied."),
    ("Pack size / MOQ", f"{PACK_SIZE} / {MIN_ORDER_QTY}",
     "No supplier data; orders are rounded to whole units with no minimum."),
    ("Stockouts in history", "Not identifiable",
     "Sales are treated as demand. If stockouts occurred, demand and safety stock are understated."),
    ("Promotions", "Not available",
     "Only a past price-change proxy is used as a feature."),
]
