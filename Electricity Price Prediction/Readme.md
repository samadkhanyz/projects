# Electricity Price Forecasting System (SMARD Data)
A complete end-to-end data science pipeline for predicting German day-ahead electricity prices. This project processes raw energy market data, performs Exploratory Data Analysis (EDA), and compares multiple statistical and machine learning models for time-series forecasting.

## Overview
The system analyzes the relationship between electricity demand, renewable generation (Wind/Solar), and market prices. It addresses challenges like seasonality, autocorrelation, and the Merit Order Effect using a variety of forecasting techniques. The dataset can be found at [Smard](https://www.smard.de/home).

## Features
* **Data Preprocessing**: Cleans and merges SMARD (ENTSO-E) CSV files, handles missing values, and resamples data to hourly intervals.
* **Feature Engineering**: Cyclical encoding (Sine/Cosine) for hours and months.
    * Lagged features (24h, 48h, 168h) for prices and load.
    * Rolling statistics and Net Load forecasting.
* **EDA Module**: Generates 5 distinct plots including correlation heatmaps and Merit Order scatter plots.
* **Multi-Model Forecasting**: Compares Naive, ARIMA, SARIMA, Facebook Prophet, and LightGBM.
* **Inference Engine**: A dedicated script to forecast prices for specific future dates using an ensemble approach.

## Project Structure
├── smard_data/               # Directory for raw CSV files from SMARD.de \
├── dataPreprocessing.py      # Cleans data and generates 'master_electricity_data.csv' \
├── modelTraining.py          # Trains all models and generates performance comparisons \
├── modelPredictions.py       # Forecasting tool for future date predictions \
├── eda_*.png                 # Generated analysis plots \
└── model_comparison_*.png    # Model performance visualizations

## Installation & Setup
### A. Install Dependencies
pip install -r requirements.txt

    pandas
    numpy
    matplotlib
    seaborn
    altair
    statsmodels
    pmdarima
    prophet
    lightgbm
    scikit-learn

### B. Data Requirements
Place your SMARD CSV files in a folder named /smard_data. The script expects files for:

* Wholesale prices
* Day-ahead forecasts
* Realized generation
* Realized load

## Usage Workflow
### 1. Data Preparation & EDA
Run this script to process the raw data and generate initial market analysis plots. \
***python dataPreprocessing.py***

### 2. Model Benchmarking
Compare the performance of different algorithms on your historical data. \
***python modelTraining.py***

Outputs: **model_comparison_table.png** and **model_comparison_predictions.png**.

### 3. Future Forecasting
Predict the electricity price for a specific date (e.g., January 27, 2026). \
***python modelPredictions.py***

## Evaluation Metrics
The models are evaluated based on:
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)

| Model | Complexity | Best For |
| :--- | :---: | :---: |
| Naive | Low | Baseline comparison |
| ARIMA/SARIMA | Medium | Capturing linear trends and 24h seasonality |
| Prophet | Medium | Handling holidays and multi-period seasonality |
| LightGBM | High | Capturing non-linear relationships (e.g., Wind vs. Price) |

**Disclaimer**: \
This project is for educational purposes. Energy markets are highly volatile and influenced by external factors (geopolitics, gas prices) not fully captured in these scripts.