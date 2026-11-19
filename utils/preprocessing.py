"""
Solar Anomaly Detection - Preprocessing Utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SELECTED_FEATURES = [
    'DC_POWER',
    'AC_POWER',
    'DAILY_YIELD',
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION'
]


def parse_datetime(series):
    """
    Robustly parse DATE_TIME series accommodating various solar dataset formats:
    - Generation data format: '%d-%m-%Y %H:%M'
    - Weather sensor format: '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d %H:%M'
    Fallback to flexible pd.to_datetime parsing.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Try common explicit formats first for speed and accuracy
    for fmt in ('%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue

    # Generic fallback
    return pd.to_datetime(series, errors='coerce')


def load_raw_datasets(generation_path, weather_path):
    """
    Load raw generation and weather CSV files and inspect preliminary stats.
    """
    df_gen = pd.read_csv(generation_path)
    df_weather = pd.read_csv(weather_path)
    return df_gen, df_weather


def preprocess_and_merge(df_gen, df_weather):
    """
    Parse dates, clean data, and merge generation data with weather sensor data
    on DATE_TIME using an inner join (matching Kaggle Solar Power Dataset structure).
    
    Returns:
        merged_df: pd.DataFrame with merged sensor and generation readings
        stats: dict containing preprocessing metadata
    """
    gen = df_gen.copy()
    weather = df_weather.copy()

    # Normalize DATE_TIME
    if 'DATE_TIME' in gen.columns:
        gen['DATE_TIME'] = parse_datetime(gen['DATE_TIME'])
    else:
        raise ValueError("Column 'DATE_TIME' missing from generation dataset.")

    if 'DATE_TIME' in weather.columns:
        weather['DATE_TIME'] = parse_datetime(weather['DATE_TIME'])
    else:
        raise ValueError("Column 'DATE_TIME' missing from weather dataset.")

    # Drop null timestamps
    gen = gen.dropna(subset=['DATE_TIME'])
    weather = weather.dropna(subset=['DATE_TIME'])

    # Merge datasets on DATE_TIME
    # Suffixes handle redundant PLANT_ID / SOURCE_KEY if present
    merged_df = pd.merge(gen, weather, on='DATE_TIME', how='inner', suffixes=('_gen', '_weather'))

    # Check for selected features
    missing_features = [f for f in SELECTED_FEATURES if f not in merged_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in merged data: {missing_features}")

    # Initial shape
    initial_len = len(merged_df)

    # Clean missing and infinite values in selected features
    for col in SELECTED_FEATURES:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(subset=SELECTED_FEATURES, inplace=True)

    stats = {
        'generation_rows': len(df_gen),
        'weather_rows': len(df_weather),
        'merged_rows_initial': initial_len,
        'merged_rows_final': len(merged_df),
        'removed_rows': initial_len - len(merged_df),
        'features': SELECTED_FEATURES
    }

    return merged_df, stats


def prepare_features(df, features=None):
    """
    Extract the selected feature DataFrame for model training / inference.
    """
    if features is None:
        features = SELECTED_FEATURES
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from dataframe: {missing}")

    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return X


def scale_features(X, scaler=None, fit=True):
    """
    Standardize features using StandardScaler (required for DBSCAN distance metrics).
    
    Args:
        X: pd.DataFrame or np.ndarray
        scaler: existing StandardScaler instance (optional)
        fit: bool, whether to fit the scaler
        
    Returns:
        X_scaled: np.ndarray of scaled values
        scaler: fitted StandardScaler
    """
    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, scaler


def validate_input_data(df, required_features=None):
    """
    Validate and prepare custom user-uploaded data for anomaly prediction.
    """
    if required_features is None:
        required_features = SELECTED_FEATURES

    missing = [f for f in required_features if f not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}", None

    clean_df = df[required_features].copy()
    for col in required_features:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

    if clean_df.isnull().all().any():
        return False, "One or more columns contain only invalid/non-numeric values.", None

    clean_df.fillna(clean_df.median(), inplace=True)
    return True, "Data is valid.", clean_df
