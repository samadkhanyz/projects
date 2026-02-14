import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import altair as alt

# --- 1. CONFIGURATION ---
BASE_DIR = os.getcwd() 
DATA_DIR = os.path.join(BASE_DIR, 'smard_data')

FILES = {
    'price':'Gro_handelspreise_202101010000_202601010000_Viertelstunde.csv',
    'forecast':'Prognostizierte_Erzeugung_Day-Ahead_202101010000_202601010000_Viertelstunde_Stunde.csv',
    'gen':'Realisierte_Erzeugung_202101010000_202601010000_Viertelstunde.csv',
    'load':'Realisierter_Stromverbrauch_202101010000_202601010000_Viertelstunde (1).csv'
}

# --- 2. MODULAR FUNCTIONS ---

def load_smard_csv(file_name):
    """
    Loads and cleans SMARD-specific CSV formatting.
    """
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path, sep=';', decimal=',', thousands='.')
    
    # Convert timestamps and set index
    df['timestamp'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M')
    df = df.set_index('timestamp').drop(['Datum von', 'Datum bis'], axis=1)
    
    # Clean numeric data (replace SMARD '-' with NaN)
    df = df.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
    
    # Resample to Hourly (Standard for Day-Ahead Markets)
    return df.resample('h').mean()

def create_generalized_dataset():
    """
    Merges all files and creates a base feature set.
    """
    print("Loading datasets...")
    df_price = load_smard_csv(FILES['price'])
    df_forecast = load_smard_csv(FILES['forecast'])
    df_load = load_smard_csv(FILES['load'])
    df_gen = load_smard_csv(FILES['gen'])

    # Select relevant columns and rename for a clean API
    data = pd.DataFrame(index=df_price.index)
    data['target_price'] = df_price['Deutschland/Luxemburg [€/MWh] Originalauflösungen']
    
    # Forecasts (Known in advance - No shift needed for renewable forecasts)
    data['fc_solar'] = df_forecast['Photovoltaik [MWh] Originalauflösungen']
    data['fc_wind_on'] = df_forecast['Wind Onshore [MWh] Originalauflösungen']
    data['fc_wind_off'] = df_forecast['Wind Offshore [MWh] Originalauflösungen']
    
    # Total renewable forecast
    data['fc_renewables_total'] = data['fc_solar'] + data['fc_wind_on'] + data['fc_wind_off']
    
    # Demand and Conventional (Historical - Shifted by 24h to avoid leaking future data)
    data['load_lag_24h'] = df_load['Netzlast [MWh] Originalauflösungen'].shift(24)
    data['load_lag_168h'] = df_load['Netzlast [MWh] Originalauflösungen'].shift(168)
    data['gen_lignite_lag_24h'] = df_gen['Braunkohle [MWh] Originalauflösungen'].shift(24)
    data['gen_gas_lag_24h'] = df_gen['Erdgas [MWh] Originalauflösungen'].shift(24)
    
    # Target Lags (Crucial for Time Series)
    data['price_lag_24h'] = data['target_price'].shift(24)
    data['price_lag_168h'] = data['target_price'].shift(168)  # 1 week ago
    data['price_lag_48h'] = data['target_price'].shift(48)    # 2 days ago

    # FIXED: Net Load Forecast (using lagged load to avoid data leakage)
    # This estimates residual demand after renewables
    data['net_load_forecast'] = data['load_lag_24h'] - data['fc_renewables_total']
    
    # Rolling statistics (capture recent trends without leakage)
    data['price_rolling_mean_24h'] = data['target_price'].shift(1).rolling(24).mean()
    data['price_rolling_std_24h'] = data['target_price'].shift(1).rolling(24).std()
    data['price_rolling_mean_168h'] = data['target_price'].shift(1).rolling(168).mean()
    
    # Calendar Features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for hour (helps capture daily patterns)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    # Cyclical encoding for month (helps capture seasonal patterns)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    return data.dropna()

def perform_eda(df):
    """
    Generates complete EDA plots for understanding market dynamics.
    """
    print("Performing EDA...")
    sns.set_style("whitegrid")
    
    # 1. Price vs Net Load Correlation
    plt.figure(figsize=(10, 6))

    plt.scatter(df['net_load_forecast'], df['target_price'], alpha=0.1, color='teal')
    plt.title('The Merit Order Effect: Price vs Forecasted Net Load')
    plt.xlabel('Net Load (Demand - Renewables) [MWh]')
    plt.ylabel('Price (€/MWh)')
    plt.tight_layout()
    plt.savefig('eda_merit_order.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Seasonality: Hourly Price Boxplots
    plt.figure(figsize=(12, 6))

    sns.boxplot(x='hour', y='target_price', data=df, palette='viridis')
    plt.title('Price Distribution by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Price (€/MWh)')
    plt.tight_layout()
    plt.savefig('eda_hourly_seasonality.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Correlation Heatmap (top features only for readability)
    plt.figure(figsize=(14, 12))

    corr_cols = ['target_price', 'fc_renewables_total', 'net_load_forecast', 
                 'price_lag_24h', 'price_lag_168h', 'load_lag_24h', 
                 'gen_lignite_lag_24h', 'gen_gas_lag_24h', 'hour', 'month']
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='RdYlGn', fmt='.2f', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Price Volatility over Time
    plt.figure(figsize=(15, 5))

    df['target_price'].tail(720).plot(linewidth=0.8, color='darkblue')
    plt.title('Electricity Price Trend (Last 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (€/MWh)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_price_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. NEW: Renewable Generation Impact
    plt.figure(figsize=(12, 6))
    
    plt.scatter(df['fc_renewables_total'], df['target_price'], alpha=0.1, color='green')
    plt.title('Impact of Renewable Generation on Prices')
    plt.xlabel('Total Renewable Forecast [MWh]')
    plt.ylabel('Price (€/MWh)')
    plt.tight_layout()
    plt.savefig('eda_renewables_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("EDA complete! Generated 5 plots.")

# --- 3. EXECUTION ---

if __name__ == "__main__":
    # Create the master dataset
    master_df = create_generalized_dataset()
    
    # Run EDA
    perform_eda(master_df)
    
    # Save the processed data
    master_df.to_csv('master_electricity_data.csv')
    
    print("\n" + "="*60)
    print("Process Complete!")
    print("="*60)
    print(f"Full Dataset Shape: {master_df.shape}")