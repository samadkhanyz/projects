import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import lightgbm as lgb

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TARGET_DATE = '2026-01-27'  # Target prediction date
RANDOM_STATE = 42

# --- 1. LOAD TRAINED MODELS AND DATA ---

def load_full_dataset(data_path='master_electricity_data.csv'):
    """
    Load the complete dataset.
    """
    print("Loading master dataset...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df):,} samples")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df

# --- 2. PREDICT WITH TIME SERIES MODELS (ARIMA, SARIMA) ---

def predict_with_arima_sarima(df, target_date, target_col='target_price'):
    """
    Predict using ARIMA and SARIMA models.
    """
    
    # Calculate steps ahead
    last_date = df.index[-1]
    target_datetime = pd.to_datetime(target_date)
    hours_ahead = int((target_datetime - last_date).total_seconds() / 3600)
    
    print(f"\nPredicting {hours_ahead} hours ahead (from {last_date} to {target_datetime})")
    
    if hours_ahead <= 0:
        print("Error: Target date is in the past or present!")
        return None
    
    ts_data = df[target_col]
    
    results = {}
    
    # ARIMA
    print("\n[1/2] Training ARIMA model on full dataset...")
    arima_model = ARIMA(ts_data, order=(2, 1, 2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=hours_ahead)
    results['ARIMA'] = arima_forecast.iloc[-1]  # Last prediction = target date
    print(f"ARIMA prediction for {target_date}: €{results['ARIMA']:.2f}/MWh")
    
    # SARIMA
    print("\n[2/2] Training SARIMA model on full dataset...")
    print("(This may take several minutes...)")
    sarima_model = SARIMAX(ts_data, 
                           order=(1, 1, 1), 
                           seasonal_order=(1, 1, 1, 24),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=hours_ahead)
    results['SARIMA'] = sarima_forecast.iloc[-1]
    print(f"SARIMA prediction for {target_date}: €{results['SARIMA']:.2f}/MWh")
    
    return results, arima_forecast, sarima_forecast

# --- 3. PREDICT WITH PROPHET ---

def predict_with_prophet(df, target_date, target_col='target_price'):
    """
    Predict using Prophet model.
    """
    
    print("\n[3/4] Training Prophet model on full dataset...")
    
    # Prepare data
    prophet_df = df.reset_index()[[df.index.name or 'timestamp', target_col]]
    prophet_df.columns = ['ds', 'y']
    
    # Train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_df)
    
    # Create future dataframe
    last_date = df.index[-1]
    target_datetime = pd.to_datetime(target_date)
    hours_ahead = int((target_datetime - last_date).total_seconds() / 3600)
    
    future = model.make_future_dataframe(periods=hours_ahead, freq='h')
    forecast = model.predict(future)
    
    # Get prediction for target date
    target_pred = forecast[forecast['ds'] == target_datetime]['yhat'].values[0]
    
    print(f"Prophet prediction for {target_date}: €{target_pred:.2f}/MWh")
    
    return target_pred, forecast

# --- 4. PREDICT WITH LIGHTGBM (Requires Feature Engineering) ---

def create_future_features(df, target_date, target_col='target_price'):
    """
    Create features for future prediction.
    Note: This requires assumptions about future renewable forecasts.
    """
    
    last_date = df.index[-1]
    target_datetime = pd.to_datetime(target_date)
    hours_ahead = int((target_datetime - last_date).total_seconds() / 3600)
    
    # Create future datetime index
    future_index = pd.date_range(start=last_date + timedelta(hours=1), 
                                  periods=hours_ahead, 
                                  freq='h')
    
    # Initialize future dataframe with target column placeholder
    future_df = pd.DataFrame(index=future_index)
    future_df[target_col] = np.nan  # Initialize target column
    
    # Calendar features (we know these for future dates)
    future_df['hour'] = future_df.index.hour
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    
    # Lag features (use recent historical data)
    last_prices = df[target_col].tail(168)  # Last week of prices
    
    # Initialize lag columns
    future_df['price_lag_24h'] = np.nan
    future_df['price_lag_168h'] = np.nan
    future_df['price_lag_48h'] = np.nan
    
    for i, future_hour in enumerate(future_index):
        # 24h lag
        if i >= 24:
            future_df.loc[future_hour, 'price_lag_24h'] = future_df.iloc[i - 24][target_col]
        else:
            lag_idx = len(last_prices) + i - 24
            if lag_idx >= 0 and lag_idx < len(last_prices):
                future_df.loc[future_hour, 'price_lag_24h'] = last_prices.iloc[lag_idx]
            else:
                future_df.loc[future_hour, 'price_lag_24h'] = df[target_col].iloc[-24 + i]
        
        # 168h lag
        if i >= 168:
            future_df.loc[future_hour, 'price_lag_168h'] = future_df.iloc[i - 168][target_col]
        else:
            lag_idx = len(last_prices) + i - 168
            if lag_idx >= 0:
                future_df.loc[future_hour, 'price_lag_168h'] = last_prices.iloc[lag_idx]
            else:
                # Use data from main df
                historical_idx = len(df) + i - 168
                if historical_idx >= 0:
                    future_df.loc[future_hour, 'price_lag_168h'] = df[target_col].iloc[historical_idx]
                else:
                    future_df.loc[future_hour, 'price_lag_168h'] = df[target_col].mean()
        
        # 48h lag
        if i >= 48:
            future_df.loc[future_hour, 'price_lag_48h'] = future_df.iloc[i - 48][target_col]
        else:
            lag_idx = len(last_prices) + i - 48
            if lag_idx >= 0 and lag_idx < len(last_prices):
                future_df.loc[future_hour, 'price_lag_48h'] = last_prices.iloc[lag_idx]
            else:
                future_df.loc[future_hour, 'price_lag_48h'] = df[target_col].iloc[-48 + i]
    
    # Initialize all feature columns that will be needed
    future_df['fc_solar'] = np.nan
    future_df['fc_wind_on'] = np.nan
    future_df['fc_wind_off'] = np.nan
    future_df['load_lag_24h'] = np.nan
    future_df['load_lag_168h'] = np.nan
    future_df['gen_lignite_lag_24h'] = np.nan
    future_df['gen_gas_lag_24h'] = np.nan
    
    # For renewable forecasts and load - use seasonal averages from historical data
    # (In production, you'd get these from actual forecast data)
    print("\nNote: Using historical seasonal averages for renewable forecasts and load")
    print("      (In production, use actual forecast data)")
    
    for idx in future_df.index:
        hour = idx.hour
        dow = idx.dayofweek
        month = idx.month
        
        # Get similar historical periods
        mask = (df.index.hour == hour) & (df.index.dayofweek == dow) & (df.index.month == month)
        historical_similar = df[mask].tail(30)  # Last 30 similar hours
        
        if len(historical_similar) > 0:
            future_df.loc[idx, 'fc_solar'] = historical_similar['fc_solar'].mean()
            future_df.loc[idx, 'fc_wind_on'] = historical_similar['fc_wind_on'].mean()
            future_df.loc[idx, 'fc_wind_off'] = historical_similar['fc_wind_off'].mean()
            future_df.loc[idx, 'load_lag_24h'] = historical_similar['load_lag_24h'].mean()
            future_df.loc[idx, 'load_lag_168h'] = historical_similar.get('load_lag_168h', historical_similar['load_lag_24h']).mean()
            future_df.loc[idx, 'gen_lignite_lag_24h'] = historical_similar['gen_lignite_lag_24h'].mean()
            future_df.loc[idx, 'gen_gas_lag_24h'] = historical_similar['gen_gas_lag_24h'].mean()
        else:
            # Fallback to overall averages
            future_df.loc[idx, 'fc_solar'] = df['fc_solar'].mean()
            future_df.loc[idx, 'fc_wind_on'] = df['fc_wind_on'].mean()
            future_df.loc[idx, 'fc_wind_off'] = df['fc_wind_off'].mean()
            future_df.loc[idx, 'load_lag_24h'] = df['load_lag_24h'].mean()
            future_df.loc[idx, 'load_lag_168h'] = df.get('load_lag_168h', df['load_lag_24h']).mean()
            future_df.loc[idx, 'gen_lignite_lag_24h'] = df['gen_lignite_lag_24h'].mean()
            future_df.loc[idx, 'gen_gas_lag_24h'] = df['gen_gas_lag_24h'].mean()
    
    # Derived features
    future_df['fc_renewables_total'] = future_df['fc_solar'] + future_df['fc_wind_on'] + future_df['fc_wind_off']
    future_df['net_load_forecast'] = future_df['load_lag_24h'] - future_df['fc_renewables_total']
    
    # Rolling features (use last known values)
    future_df['price_rolling_mean_24h'] = df[target_col].tail(24).mean()
    future_df['price_rolling_std_24h'] = df[target_col].tail(24).std()
    future_df['price_rolling_mean_168h'] = df[target_col].tail(168).mean()
    
    return future_df

def predict_with_lightgbm(df, target_date, target_col='target_price'):
    """
    Predict using LightGBM model.
    """
    
    print("\n[4/4] Training LightGBM model on full dataset...")
    
    # Prepare training data
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Train model
    params = {
        'num_leaves': 50,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    
    # Create future features
    future_df = create_future_features(df, target_date, target_col)
    
    # Iterative prediction (important for lag features)
    predictions = []
    
    # Ensure all required columns exist before prediction
    missing_cols = [col for col in feature_cols if col not in future_df.columns]
    if missing_cols:
        print(f"Warning: Missing features {missing_cols}, filling with 0")
        for col in missing_cols:
            future_df[col] = 0
    
    for i in range(len(future_df)):
        # Get features for this hour - ensure correct order
        X_pred = future_df.iloc[i:i+1][feature_cols]
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        
        # Update future_df with this prediction for subsequent lag calculations
        future_df.iloc[i, future_df.columns.get_loc(target_col)] = pred
    
    # Get the final prediction for target date
    target_datetime = pd.to_datetime(target_date)
    target_pred = predictions[-1]  # Last prediction is for target date
    
    print(f"LightGBM prediction for {target_date}: €{target_pred:.2f}/MWh")
    
    return target_pred, future_df, predictions

# --- 5. VISUALIZATION ---

def visualize_predictions(df, predictions_dict, target_date):
    """
    Create visualization showing historical data and predictions.
    """
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot last 30 days of historical data
    last_30_days = df.tail(720)  # 30 days * 24 hours
    ax.plot(last_30_days.index, last_30_days['target_price'], 
            label='Historical Price', color='black', linewidth=2, alpha=0.7)
    
    # Mark target date
    target_datetime = pd.to_datetime(target_date)
    
    # Plot predictions
    colors = {'ARIMA': 'blue', 'SARIMA': 'green', 'Prophet': 'red', 'LightGBM': 'orange'}
    
    for model_name, pred_value in predictions_dict.items():
        ax.scatter(target_datetime, pred_value, 
                  color=colors.get(model_name, 'purple'), 
                  s=200, marker='*', 
                  label=f'{model_name}: €{pred_value:.2f}/MWh',
                  zorder=5, edgecolors='black', linewidths=1.5)
    
    # Formatting
    ax.axvline(df.index[-1], color='gray', linestyle='--', alpha=0.5, label='Last Known Data')
    ax.axvline(target_datetime, color='red', linestyle='--', alpha=0.5, label='Target Date')
    
    ax.set_title(f'Electricity Price Predictions for {target_date}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (€/MWh)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'prediction_{target_date}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization: prediction_{target_date}.png")

def create_prediction_summary(predictions_dict, target_date):
    """
    Create a summary table of predictions.
    """
    
    summary_df = pd.DataFrame({
        'Model': list(predictions_dict.keys()),
        'Predicted Price (€/MWh)': [f"{v:.2f}" for v in predictions_dict.values()]
    })
    
    # Calculate statistics
    prices = list(predictions_dict.values())
    mean_pred = np.mean(prices)
    std_pred = np.std(prices)
    min_pred = np.min(prices)
    max_pred = np.max(prices)
    
    print("\n" + "="*70)
    print(f"ELECTRICITY PRICE PREDICTIONS FOR {target_date}")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("\n" + "-"*70)
    print(f"Ensemble Average:  €{mean_pred:.2f}/MWh")
    print(f"Standard Deviation: €{std_pred:.2f}/MWh")
    print(f"Range:             €{min_pred:.2f} - €{max_pred:.2f}/MWh")
    print("="*70)
    
    return summary_df, mean_pred

# --- 6. MAIN PREDICTION PIPELINE ---

def predict_electricity_price(target_date='2026-01-27', 
                              data_path='master_electricity_data.csv',
                              hourly_prediction=True):
    """
    Main function to predict electricity price for a future date.
    
    Parameters:
    -----------
    target_date : str
        Target date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
    data_path : str
        Path to the master dataset
    hourly_prediction : bool
        If True and no time specified, predicts for each hour of the day
    """
    
    print("\n" + "="*70)
    print("ELECTRICITY PRICE FORECASTING SYSTEM")
    print("="*70)
    
    # Load data
    df = load_full_dataset(data_path)
    
    # If only date provided (no time), predict for noon or all hours
    if len(target_date) == 10:  # Format: YYYY-MM-DD
        if hourly_prediction:
            print(f"\nGenerating hourly predictions for {target_date} (00:00 to 23:00)")
            target_date_full = f"{target_date} 12:00:00"  # Use noon for main prediction
        else:
            target_date_full = f"{target_date} 12:00:00"
    else:
        target_date_full = target_date
    
    # Run predictions
    predictions = {}
    
    # Time series models
    ts_results, arima_fc, sarima_fc = predict_with_arima_sarima(df, target_date_full)
    predictions.update(ts_results)
    
    # Prophet
    prophet_pred, prophet_fc = predict_with_prophet(df, target_date_full)
    predictions['Prophet'] = prophet_pred
    
    # LightGBM
    lgb_pred, future_features, lgb_preds = predict_with_lightgbm(df, target_date_full)
    predictions['LightGBM'] = lgb_pred
    
    # Create summary
    summary_df, ensemble_pred = create_prediction_summary(predictions, target_date_full)
    
    # Visualize
    visualize_predictions(df, predictions, target_date_full)
    
    print("Prediction complete!")
    print(f"Recommended Ensemble Prediction: €{ensemble_pred:.2f}/MWh")
    
    return predictions, ensemble_pred, summary_df

# --- 7. RUN PREDICTION ---

if __name__ == "__main__":
    # Predict for January 27, 2026
    predictions, ensemble, summary = predict_electricity_price(
        target_date='2026-01-27',
        data_path='master_electricity_data.csv'
    )
    
    print("\n" + "="*70)
    print("DONE! Check the generated visualization.")
    print("="*70)