# Demand Forecasting & Inventory Optimization Using Machine Learning

## Project Overview

This project aims to analyze historical retail sales data, build machine learning models to forecast product demand, and use the forecasts to support inventory planning. The goal is to reduce stock shortages and unnecessary excess inventory through data-driven decisions.

The project uses the Online Retail dataset and follows a four-week development plan, beginning with data exploration and progressing toward forecasting models, inventory calculations, and an interactive dashboard.

## Objectives

- Understand historical retail sales patterns.
- Clean and prepare the dataset for analysis.
- Explore product demand and identify trends over time.
- Develop baseline and machine learning forecasting models.
- Evaluate forecasts using appropriate performance metrics.
- Estimate inventory requirements using demand forecasts and documented assumptions.
- Present results through an interactive dashboard.

## Dataset

**Dataset:** Online Retail

**Source:** UCI Machine Learning Repository

The dataset contains transaction records from an online retail business. It includes information such as invoice numbers, product codes, product descriptions, quantities, invoice dates, unit prices, customer identifiers, and countries.

The original dataset is retained separately from processed data so that the cleaning and analysis steps can be reproduced.

## Technologies Used

- **Python:** Data processing and model development.
- **Jupyter Notebook:** Exploratory data analysis and experimentation.
- **Pandas:** Data manipulation and cleaning.
- **NumPy:** Numerical operations.
- **Matplotlib and Seaborn:** Data visualization.
- **Scikit-learn:** Forecasting models and evaluation.
- **Git and GitHub:** Version control and project management.
- **Streamlit and Plotly:** Planned dashboard development.

## Project Progress

### Week 1: Data Collection and Exploratory Data Analysis

The first week focuses on understanding the dataset, assessing data quality, and preparing the data for forecasting.

Completed analysis tasks include:

- Loading and inspecting the original dataset.
- Examining the dataset's structure, data types, and summary statistics.
- Identifying missing values and duplicate records.
- Investigating negative quantities and non-positive unit prices.
- Preparing a data dictionary and data quality report.
- Creating an interim cleaned dataset for further analysis.
- Exploring the data using visualizations.

Negative quantities may represent returns or cancellations, so they require careful consideration rather than automatic removal. Data preparation must preserve the information needed for demand forecasting.

### Week 2: Feature Engineering and Baseline Forecasting

Run `notebooks/02_week2_features_baselines.ipynb` after the Week 1 notebook. It reads `data/processed/weekly_product_demand.csv`.

**Aggregation level:** weekly × product (StockCode). The data has no store dimension.
**Product scope:** 978 products that existed for the whole train period and sold in ≥80% of train weeks. They cover 70.4% of train units, and the rule uses train data only (`reports/product_selection.csv`).
**Forecast horizons:** h = 1 (next week) and h = 4 (four weeks ahead, for replenishment). Each horizon has its own feature table (direct strategy).

**Chronological split** (`src/splits.py`, `reports/split_plan.csv`)

| Split | Weeks | Use |
|---|---|---|
| Train | 2010-12-06 → 2011-08-08 (35) | fitting |
| Validation | 2011-08-15 → 2011-10-03 (8) | tuning and model choice |
| Test | 2011-10-10 → 2011-11-28 (8) | final evaluation only |

The closed week (27 Dec 2010) and the partial weeks (29 Nov 2010, 5 Dec 2011) are removed. Lags skip over the closed week.

**Features** (`src/features.py`, `reports/feature_dictionary.csv`): lags, rolling mean/std/max, EWM, intermittency, weeks since last sale, order counts, a price-change proxy, product age, and calendar features (trading days, UK bank holidays, week, month, quarter, weeks to Christmas).
Demand features use only data up to the forecast origin (t − h). `check_no_leakage()` scrambles future values and confirms that no feature changes.

**Baselines** (`src/baselines.py`): naive, 4-week moving average, seasonal-naive (52 weeks), and a 4-week seasonal proxy.
`seasonal_naive_52` has 0% coverage because the data has no prior year.

**Initial evaluation** (`reports/baseline_evaluation.csv`): the best baseline is `moving_avg_4`.

| Horizon | Validation WAPE | Test WAPE | Test bias |
|---|---|---|---|
| h = 1 | 67.95% | 61.78% | −2.05% |
| h = 4 | 69.42% | 65.79% | −9.05% |

**Missing data and stockouts:** zeros in open weeks are treated as true zeros. Stockouts cannot be identified because the data has no inventory fields. Promotions are not available.

### Week 3: Model Development and Inventory Optimization

Planned tasks include:

- Training and comparing suitable forecasting models.
- Evaluating performance using MAE, RMSE, and an appropriate percentage-based metric.
- Comparing model performance against baseline forecasts.
- Estimating safety stock and reorder points using demand variability and lead-time assumptions.
- Documenting inventory-data limitations and assumptions.

The dataset does not directly provide all operational inventory details, such as current stock levels and supplier lead times. Any inventory recommendations will therefore require clearly documented assumptions or additional data.

### Week 4: Dashboard and Final Documentation

Planned tasks include:

- Building an interactive dashboard using Streamlit and Plotly.
- Displaying demand trends and forecast results.
- Presenting model evaluation metrics.
- Showing inventory recommendations based on the available inputs and assumptions.
- Completing the project report and documentation.
- Preparing a final project presentation.

## Project Structure

```text
Demand-Forecasting-Inventory-Optimization-Using-Machine-Learning/
│
├── data/
│   ├── interim/
│   │   └── cleaned_sales.csv
│   ├── processed/
│   └── raw/
│       └── Online Retail.xlsx
│
├── notebooks/
│   └── 01_week1_data_quality_eda.ipynb
│
├── reports/
│   ├── data_dictionary.csv
│   ├── data_quality_report.csv
|
├── src/
│   └── convert_to_csv.py
│
├── README.md
└── requirements.txt
```

This structure describes the intended organization. Folder and file names should be updated to match the actual repository contents.

## Evaluation Metrics

Forecasting models will be evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted demand.
- **Root Mean Squared Error (RMSE):** Penalizes larger forecasting errors more heavily.
- **Percentage-based metric:** Used when suitable for the demand data, with care for zero or very low actual demand.

## Expected Outcome

The expected outcome is a reproducible demand forecasting workflow that compares forecasting models, evaluates their accuracy, and demonstrates how demand estimates can support inventory planning. An interactive dashboard will make the results easier to understand and use.

## Current Status

Week 1 focuses on data quality assessment, cleaning, and exploratory analysis. Forecasting models, inventory optimization, and dashboard development are planned for the subsequent weeks.

## How to Run

1. Clone or download the repository.
2. Install the required Python libraries.
3. Place the original dataset in the configured raw-data directory.
4. Open the Week 1 notebook in Jupyter Notebook or VS Code.
5. Run the notebook cells in order.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Ensure that the dataset path in the notebook matches your local folder structure.

## Limitations

- Historical transactions do not directly provide complete inventory and supplier lead-time information.
- Returns and cancellations must be handled carefully when estimating demand.
- Forecast quality depends on the amount, quality, and time coverage of the available data.
- Inventory recommendations depend on the assumptions and operational information available.

## Future Improvements

- Incorporate additional inventory and supplier data.
- Evaluate more advanced forecasting techniques.
- Improve inventory recommendations using actual lead times and stock levels.
- Add interactive product-level filtering and forecast comparisons to the dashboard.
