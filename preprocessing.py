import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path='data/data.csv'):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def convert_numeric_columns(df):
    """Convert specified columns to numeric type."""
    numeric_columns = [
        'zone1_temperature', 'zone2_temperature',
        'equipment_energy_consumption', 'zone1_humidity',
        'lighting_energy'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'temperature' in col or 'humidity' in col:
            df[col] = df[col].astype(float)
    
    return df

def get_zone_features(df):
    """Extract temperature and humidity features for all zones."""
    temperature_cols = [f'zone{i}_temperature' for i in range(1, 10)]
    humidity_cols = [f'zone{i}_humidity' for i in range(1, 10)]
    
    return df[temperature_cols], df[humidity_cols]

def fill_missing_values(df):
    """Fill missing values in all columns with median values."""
    # Fill zone humidity values
    for i in range(1, 10):
        df[f'zone{i}_humidity'] = df[f'zone{i}_humidity'].fillna(df[f'zone{i}_humidity'].median())
        df[f'zone{i}_temperature'] = df[f'zone{i}_temperature'].fillna(df[f'zone{i}_temperature'].median())
    
    # Fill other columns
    columns_to_fill = [
        'lighting_energy', 'outdoor_temperature', 'atmospheric_pressure',
        'wind_speed', 'outdoor_humidity', 'visibility_index', 'dew_point',
        'random_variable1', 'random_variable2', 'equipment_energy_consumption'
    ]
    
    for col in columns_to_fill:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df

def create_heat_features(df):
    """Create heat features by multiplying temperature and humidity for each zone."""
    for i in range(1, 10):
        df[f'zone{i}_heat'] = df[f'zone{i}_temperature'] * df[f'zone{i}_humidity']
    
    # Drop original temperature and humidity columns
    cols_to_drop = []
    for i in range(1, 10):
        cols_to_drop.extend([f'zone{i}_temperature', f'zone{i}_humidity'])
    
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def add_time_features(df):
    """Add time-based features to the dataframe."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)  
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday 
    df['month'] = df.index.month
    
    # Add month difference feature
    df['First_difference_month'] = df['month'] - df['month'].shift(1)
    df.dropna(subset=['First_difference_month'], inplace=True)
    df.drop(columns=['month'], inplace=True)
    
    return df

def scale_features(df):
    """Scale features using MinMaxScaler."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def preprocess_data(file_path='data/data.csv'):
    """Main preprocessing pipeline."""
    # Load data
    df = load_data(file_path)
    print('Data loaded successfully')
    # Convert numeric columns
    df = convert_numeric_columns(df)
    print('Numeric columns converted successfully')
    # Fill missing values
    df = fill_missing_values(df)
    print('Missing values filled successfully')
    # Create heat features
    df = create_heat_features(df)
    print('Heat features created successfully') 
    # Add time features
    df = add_time_features(df)
    print('Time features added successfully')
    # Scale features
    scaled_data = scale_features(df)
    print('Features scaled successfully')
    
    return df, scaled_data

data, scaled_data = preprocess_data() 