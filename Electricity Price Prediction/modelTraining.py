'''
Baseline Models

- Naive
- ARIMA
- SARIMA
- Prophet

Model Training
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from datetime import datetime

# Statistical Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Prophet
from prophet import Prophet

# LightGBM
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- 1. DATA PREPARATION ---

def create_train_val_test_split(df, train_size=0.7, val_size=0.15):
    """
    Creates chronological train/val/test split for time series.
    Default: 70% train, 15% val, 15% test
    """
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    print("="*70)
    print("DATASET SPLIT SUMMARY")
    print("="*70)
    print(f"Train: {train.index[0]} to {train.index[-1]} ({len(train):,} samples, {len(train)/n*100:.1f}%)")
    print(f"Val:   {val.index[0]} to {val.index[-1]} ({len(val):,} samples, {len(val)/n*100:.1f}%)")
    print(f"Test:  {test.index[0]} to {test.index[-1]} ({len(test):,} samples, {len(test)/n*100:.1f}%)")
    print("="*70 + "\n")
    
    return train, val, test

def prepare_data_for_models(train, val, test, target_col='target_price'):
    """Prepares datasets for different model types."""
    # For tree-based models (LightGBM)
    feature_cols = [col for col in train.columns if col != target_col]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    
    X_val = val[feature_cols]
    y_val = val[target_col]
    
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # For time series models (ARIMA, SARIMA, Prophet)
    ts_train = train[target_col]
    ts_val = val[target_col]
    ts_test = test[target_col]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'ts_train': ts_train,
        'ts_val': ts_val,
        'ts_test': ts_test
    }

# --- 2. EVALUATION METRICS ---

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# --- 3. BASELINE MODELS ---

def naive_baseline(data_dict):
    """Naive forecast: Use yesterday's price (24h lag)."""
    print("\n" + "="*70)
    print("MODEL 1: NAIVE BASELINE (24h Lag)")
    print("="*70)
    
    start_time = time.time()
    
    # Prediction is simply the price 24 hours ago
    y_pred_val = data_dict['ts_train'].iloc[-len(data_dict['ts_val']):].values
    y_pred_test = data_dict['ts_val'].iloc[-len(data_dict['ts_test']):].values
    
    elapsed = time.time() - start_time
    
    metrics_val = calculate_metrics(data_dict['y_val'], y_pred_val, "Naive - Val")
    metrics_test = calculate_metrics(data_dict['y_test'], y_pred_test, "Naive - Test")
    
    print(f"Training Time: {elapsed:.2f}s")
    print(f"Val  → RMSE: {metrics_val['RMSE']:.2f}, MAE: {metrics_val['MAE']:.2f}, MAPE: {metrics_val['MAPE']:.2f}%")
    print(f"Test → RMSE: {metrics_test['RMSE']:.2f}, MAE: {metrics_test['MAE']:.2f}, MAPE: {metrics_test['MAPE']:.2f}%")
    
    return {
        'model': None,
        'predictions_val': y_pred_val,
        'predictions_test': y_pred_test,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'training_time': elapsed
    }

# --- 4. ARIMA MODEL ---

def train_arima(data_dict, order=(2, 1, 2)):
    """Train ARIMA model with specified order."""
    print("\n" + "="*70)
    print(f"MODEL 2: ARIMA{order}")
    print("="*70)
    
    start_time = time.time()
    
    # Fit on training data
    print("Fitting ARIMA model...")
    model = ARIMA(data_dict['ts_train'], order=order)
    model_fit = model.fit()
    
    # Forecast for validation period
    y_pred_val = model_fit.forecast(steps=len(data_dict['ts_val']))
    
    # Refit on train + val for test predictions
    ts_train_val = pd.concat([data_dict['ts_train'], data_dict['ts_val']])
    model_test = ARIMA(ts_train_val, order=order)
    model_test_fit = model_test.fit()
    y_pred_test = model_test_fit.forecast(steps=len(data_dict['ts_test']))
    
    elapsed = time.time() - start_time
    
    metrics_val = calculate_metrics(data_dict['y_val'], y_pred_val, "ARIMA - Val")
    metrics_test = calculate_metrics(data_dict['y_test'], y_pred_test, "ARIMA - Test")
    
    print(f"Training Time: {elapsed:.2f}s")
    print(f"Val  → RMSE: {metrics_val['RMSE']:.2f}, MAE: {metrics_val['MAE']:.2f}, MAPE: {metrics_val['MAPE']:.2f}%")
    print(f"Test → RMSE: {metrics_test['RMSE']:.2f}, MAE: {metrics_test['MAE']:.2f}, MAPE: {metrics_test['MAPE']:.2f}%")
    
    return {
        'model': model_fit,
        'predictions_val': y_pred_val,
        'predictions_test': y_pred_test,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'training_time': elapsed
    }

# --- 5. SARIMA MODEL ---

def train_sarima(data_dict, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    """Train SARIMA model (24h seasonality for hourly data)."""
    print("\n" + "="*70)
    print(f"MODEL 3: SARIMA{order}x{seasonal_order}")
    print("="*70)
    print("Warning: This may take several minutes...")
    
    start_time = time.time()
    
    # Fit on training data
    print("Fitting SARIMA model...")
    model = SARIMAX(data_dict['ts_train'], 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Forecast for validation
    y_pred_val = model_fit.forecast(steps=len(data_dict['ts_val']))
    
    # Refit on train + val for test
    ts_train_val = pd.concat([data_dict['ts_train'], data_dict['ts_val']])
    model_test = SARIMAX(ts_train_val, 
                         order=order, 
                         seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    model_test_fit = model_test.fit(disp=False)
    y_pred_test = model_test_fit.forecast(steps=len(data_dict['ts_test']))
    
    elapsed = time.time() - start_time
    
    metrics_val = calculate_metrics(data_dict['y_val'], y_pred_val, "SARIMA - Val")
    metrics_test = calculate_metrics(data_dict['y_test'], y_pred_test, "SARIMA - Test")
    
    print(f"Training Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"Val  → RMSE: {metrics_val['RMSE']:.2f}, MAE: {metrics_val['MAE']:.2f}, MAPE: {metrics_val['MAPE']:.2f}%")
    print(f"Test → RMSE: {metrics_test['RMSE']:.2f}, MAE: {metrics_test['MAE']:.2f}, MAPE: {metrics_test['MAPE']:.2f}%")
    
    return {
        'model': model_fit,
        'predictions_val': y_pred_val,
        'predictions_test': y_pred_test,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'training_time': elapsed
    }

# --- 6. PROPHET MODEL ---

def train_prophet(train, val, test, target_col='target_price'):
    """Train Facebook Prophet model."""
    print("\n" + "="*70)
    print("MODEL 4: PROPHET")
    print("="*70)
    
    start_time = time.time()
    
    # Prepare data for Prophet
    prophet_train = train.reset_index()[['timestamp', target_col]]
    prophet_train.columns = ['ds', 'y']
    
    # Train model
    print("Fitting Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(prophet_train)
    
    # Validation predictions
    future_val = pd.DataFrame({'ds': val.index})
    forecast_val = model.predict(future_val)
    y_pred_val = forecast_val['yhat'].values
    
    # Refit on train + val for test
    prophet_train_val = pd.concat([train, val]).reset_index()[['timestamp', target_col]]
    prophet_train_val.columns = ['ds', 'y']
    model_test = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model_test.fit(prophet_train_val)
    future_test = pd.DataFrame({'ds': test.index})
    forecast_test = model_test.predict(future_test)
    y_pred_test = forecast_test['yhat'].values
    
    elapsed = time.time() - start_time
    
    metrics_val = calculate_metrics(val[target_col], y_pred_val, "Prophet - Val")
    metrics_test = calculate_metrics(test[target_col], y_pred_test, "Prophet - Test")
    
    print(f"Training Time: {elapsed:.2f}s")
    print(f"Val  → RMSE: {metrics_val['RMSE']:.2f}, MAE: {metrics_val['MAE']:.2f}, MAPE: {metrics_val['MAPE']:.2f}%")
    print(f"Test → RMSE: {metrics_test['RMSE']:.2f}, MAE: {metrics_test['MAE']:.2f}, MAPE: {metrics_test['MAPE']:.2f}%")
    
    return {
        'model': model,
        'predictions_val': y_pred_val,
        'predictions_test': y_pred_test,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'training_time': elapsed
    }

# --- 7. LIGHTGBM WITH HYPERPARAMETER TUNING ---

def train_lightgbm(data_dict, tune_hyperparams=True):
    """Train LightGBM with optional hyperparameter tuning."""
    print("\n" + "="*70)
    print("MODEL 5: LIGHTGBM" + (" (with Hyperparameter Tuning)" if tune_hyperparams else ""))
    print("="*70)
    
    start_time = time.time()
    
    if tune_hyperparams:
        print("Tuning hyperparameters using validation set...")
        
        # Hyperparameter search space
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 300, 500],
            'max_depth': [-1, 10, 20],
            'min_child_samples': [20, 50, 100]
        }
        
        best_score = float('inf')
        best_params = None
        
        # Grid search (simplified - you can use optuna for better search)
        from itertools import product
        
        param_combinations = [
            {'num_leaves': 50, 'learning_rate': 0.05, 'n_estimators': 300, 
             'max_depth': 10, 'min_child_samples': 20},
            {'num_leaves': 70, 'learning_rate': 0.01, 'n_estimators': 500, 
             'max_depth': -1, 'min_child_samples': 50},
            {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100, 
             'max_depth': 20, 'min_child_samples': 100}
        ]
        
        for params in param_combinations:
            model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
            model.fit(data_dict['X_train'], data_dict['y_train'])
            y_pred = model.predict(data_dict['X_val'])
            score = mean_squared_error(data_dict['y_val'], y_pred)
            
            if score < best_score:
                best_score = score
                best_params = params
        
        print(f"Best params: {best_params}")
        print(f"Best validation RMSE: {np.sqrt(best_score):.2f}")
        
        final_params = best_params
    else:
        final_params = {
            'num_leaves': 50,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 10,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
    
    # Train final model on train set
    print("Training final model...")
    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        data_dict['X_train'], 
        data_dict['y_train'],
        eval_set=[(data_dict['X_val'], data_dict['y_val'])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Predictions
    y_pred_val = model.predict(data_dict['X_val'])
    
    # Retrain on train + val for test
    X_train_val = pd.concat([data_dict['X_train'], data_dict['X_val']])
    y_train_val = pd.concat([data_dict['y_train'], data_dict['y_val']])
    
    model_final = lgb.LGBMRegressor(**final_params)
    model_final.fit(X_train_val, y_train_val)
    y_pred_test = model_final.predict(data_dict['X_test'])
    
    elapsed = time.time() - start_time
    
    metrics_val = calculate_metrics(data_dict['y_val'], y_pred_val, "LightGBM - Val")
    metrics_test = calculate_metrics(data_dict['y_test'], y_pred_test, "LightGBM - Test")
    
    print(f"Training Time: {elapsed:.2f}s")
    print(f"Val  → RMSE: {metrics_val['RMSE']:.2f}, MAE: {metrics_val['MAE']:.2f}, MAPE: {metrics_val['MAPE']:.2f}%")
    print(f"Test → RMSE: {metrics_test['RMSE']:.2f}, MAE: {metrics_test['MAE']:.2f}, MAPE: {metrics_test['MAPE']:.2f}%")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': data_dict['X_train'].columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_imp.head(10).to_string(index=False))
    
    return {
        'model': model_final,
        'predictions_val': y_pred_val,
        'predictions_test': y_pred_test,
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'training_time': elapsed,
        'feature_importance': feature_imp
    }

# --- 8. VISUALIZATION ---

def plot_results(results, val, test, target_col='target_price'):
    """Create comprehensive visualization of all model predictions."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Validation Set
    ax = axes[0]
    ax.plot(val.index, val[target_col], label='Actual', color='black', linewidth=2, alpha=0.7)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for (name, result), color in zip(results.items(), colors):
        ax.plot(val.index, result['predictions_val'], label=name, 
                color=color, alpha=0.6, linewidth=1.5)
    
    ax.set_title('Validation Set Predictions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (€/MWh)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Test Set
    ax = axes[1]
    ax.plot(test.index, test[target_col], label='Actual', color='black', linewidth=2, alpha=0.7)
    
    for (name, result), color in zip(results.items(), colors):
        ax.plot(test.index, result['predictions_test'], label=name, 
                color=color, alpha=0.6, linewidth=1.5)
    
    ax.set_title('Test Set Predictions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (€/MWh)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: model_comparison_predictions.png")

def create_comparison_table(results):
    """Create performance comparison table."""
    
    metrics_list = []
    for name, result in results.items():
        metrics_list.append(result['metrics_val'])
        metrics_list.append(result['metrics_test'])
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Create styled table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_metrics.values, 
                     colLabels=df_metrics.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color coding
    for i in range(len(df_metrics)):
        if 'Val' in df_metrics.iloc[i]['Model']:
            for j in range(len(df_metrics.columns)):
                table[(i+1, j)].set_facecolor('#E8F4F8')
        else:
            for j in range(len(df_metrics.columns)):
                table[(i+1, j)].set_facecolor('#FFF4E6')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_table.png")
    
    return df_metrics

# --- 9. MAIN EXECUTION ---

def run_complete_experiment(data_path='master_electricity_data.csv'):
    """Run complete model comparison experiment."""
    
    print("\n" + "="*70)
    print("ELECTRICITY PRICE FORECASTING - COMPLETE MODEL COMPARISON")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df):,} samples with {df.shape[1]} features")
    
    # Create splits
    train, val, test = create_train_val_test_split(df, train_size=0.7, val_size=0.15)
    
    # Prepare data
    data_dict = prepare_data_for_models(train, val, test)
    
    # Store results
    results = {}
    
    # Model 1: Naive Baseline
    results['Naive'] = naive_baseline(data_dict)
    
    # Model 2: ARIMA
    results['ARIMA'] = train_arima(data_dict, order=(2, 1, 2))
    
    # Model 3: SARIMA (may take time!)
    results['SARIMA'] = train_sarima(data_dict, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    
    # Model 4: Prophet
    results['Prophet'] = train_prophet(train, val, test)
    
    # Model 5: LightGBM
    results['LightGBM'] = train_lightgbm(data_dict, tune_hyperparams=True)
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    plot_results(results, val, test)
    comparison_df = create_comparison_table(results)
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\nTest Set Performance:")
    print(comparison_df[comparison_df['Model'].str.contains('Test')].to_string(index=False))
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("Output files generated:")
    print("  - model_comparison_predictions.png")
    print("  - model_comparison_table.png")
    
    return results, comparison_df

# --- RUN EXPERIMENT ---
if __name__ == "__main__":
    results, metrics_df = run_complete_experiment('master_electricity_data.csv')