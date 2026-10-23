# Demand Forecasting and Inventory Optimization Using Machine Learning

## Project Overview

This project aims to develop a machine learning-based system for forecasting retail demand and supporting inventory optimization. By analyzing historical retail sales data, the system will work toward predicting future demand and generating data-driven inventory recommendations to help reduce stockouts and overstocking.

## Work Completed

The initial data inspection and validation phase has started. The following work has been completed:

- **Online Retail Dataset (`Online Retail.xlsx`):** The original Excel dataset is available as the primary data source for the project.
- **CSV Conversion Script (`convert_to_csv.py`):** A Python utility using Pandas is available for converting Excel data into CSV format.
- **Data Loading and Inspection:** The dataset has been loaded into a Pandas DataFrame, and its columns, data types, and initial records have been inspected.
- **Initial Data Quality Report:** Missing values, unique values, and exact duplicate rows have been examined.
- **Data Validation Checks:** Initial checks have been performed to identify cancellation invoices, negative and zero quantities, non-positive prices, missing product descriptions, and missing customer IDs.
- **Data Type Verification:** The transaction date and numerical fields have been checked to establish their data types.

The original dataset contains 541,909 records and 8 columns. At this stage, the project is focused on understanding the data and identifying potential quality issues before applying cleaning rules.

## Upcoming Work

- Standardization and validation of data fields.
- Handling missing values and duplicate records using documented rules.
- Reviewing returns, cancellations, and invalid prices.
- Preparing a cleaned dataset and data dictionary.
- Exploratory data analysis (EDA) to identify demand trends and patterns.
- Feature engineering and preparation of time-series data.
- Development and evaluation of demand forecasting models.
- Inventory optimization and replenishment recommendations.
- Visualization of forecasts and inventory insights.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab for notebook-based development

## Project Status

**Phase 1: Data Collection and Initial Data Validation — In Progress**

The initial inspection has been completed through the data validation checks. Further cleaning and analysis are pending. Development is currently paused due to a Google Colab runtime connection issue.

The repository will be updated as the project progresses through data cleaning, exploratory analysis, model development, evaluation, and application development.
