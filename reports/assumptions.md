# Assumptions and limitations (Week 3)

All inventory parameters are **assumptions**. None of them comes from the dataset. They are set in `src/config.py`.

| Assumption | Value used | Notes |
|---|---|---|
| Lead time | 2 weeks for every product | No supplier data. Configurable from 1 to 4 weeks (the model horizons). |
| Review period | 1 week | Orders are placed weekly (periodic review, order-up-to policy). |
| Service level | 95% cycle service level | Probability of not running out during lead time + review period. |
| Demand variability | RMSE of out-of-sample cumulative forecast errors per product | Measured on the validation period with models trained on the train period only. |
| Error distribution | Normal approximation | z = NormalDist().inv_cdf(service level). Approximate for intermittent items. |
| Current stock | User input only | Not in the dataset. Outputs that need it stay empty until stock is supplied. |
| Pack size / MOQ | 1 / 0 | No supplier data; orders are rounded to whole units with no minimum. |
| Stockouts in history | Not identifiable | Sales are treated as demand. If stockouts occurred, demand and safety stock are understated. |
| Promotions | Not available | Only a past price-change proxy is used as a feature. |

## Model limitations

- **Only about 12 months of data:** yearly seasonality cannot be learned. Validation and test fall in the autumn ramp-up and the pre-Christmas peak.
- **Selection rule:** the selected method (`ensemble`) was chosen on validation, using the lowest mean WAPE with |bias| ≤ 10.0%. The test period was used once, for the final comparison.
- **Noisy product-level demand:** product-level WAPE stays high (about 60–70%) because weekly demand per product is noisy and often intermittent. Total-level error is much lower.
- **Hidden stockouts:** the data has no stockout or inventory history, so if stockouts happened, the forecasts and safety stock are understated.
- **No promotion data:** promotions cannot be modelled.
- **Christmas week:** the 26 Dec 2011 forecast assumes the shop trades 4 days. In 2010 it was closed that week.
- **Normal approximation:** the safety stock uses a normal approximation, which is less accurate for intermittent items. The test backtest in `reports/service_level_backtest.csv` shows the service level actually achieved.
- **Service level below target:** on the test period, a 95% target achieved 91.0%. Setting 99% achieved about 94%. The variability estimate comes from only 7 validation cycles per product, and peak-season risk is higher.
- **Products being phased out:** for products whose demand collapses, forecasts lag the drop by several weeks. Flag products with `trend_4_vs_12 < 0.5` for manual review.