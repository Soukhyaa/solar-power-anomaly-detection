"""
Solar Anomaly Detection - Prediction Utilities
Provides inference interfaces for single instances and batch datasets.
"""

import os
import joblib
import pandas as pd
import numpy as np
from .preprocessing import SELECTED_FEATURES, validate_input_data


def load_model(model_path):
    """
    Load a trained scikit-learn model or scaler using joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def predict_isolation_forest(model, X):
    """
    Run inference with Isolation Forest.
    
    Returns:
        preds: np.ndarray (-1 for anomaly, 1 for normal)
        labels: list of strings ('Anomaly' or 'Normal')
        scores: np.ndarray (anomaly score, lower/negative indicates more anomalous)
    """
    preds = model.predict(X)
    # IsolationForest.decision_function: average anomaly score of the base classifiers.
    # The lower, the more abnormal. Negative scores represent anomalies.
    scores = model.decision_function(X)
    labels = ["Anomaly" if p == -1 else "Normal" for p in preds]
    return preds, labels, scores


def predict_single_instance(model, feature_values):
    """
    Predict anomaly status for a single observation dictionary or series.
    
    Args:
        model: Trained IsolationForest instance
        feature_values: dict or pd.Series with keys matching SELECTED_FEATURES
        
    Returns:
        result: dict with prediction ('Normal' or 'Anomaly'), score, and feature echoes
    """
    row_data = {}
    for feat in SELECTED_FEATURES:
        if feat not in feature_values:
            raise ValueError(f"Missing required feature: {feat}")
        row_data[feat] = float(feature_values[feat])

    df_single = pd.DataFrame([row_data])
    preds, labels, scores = predict_isolation_forest(model, df_single)
    
    is_anomaly = bool(preds[0] == -1)
    
    return {
        'prediction': str(labels[0]),
        'is_anomaly': bool(is_anomaly),
        'anomaly_code': int(preds[0]),
        'anomaly_score': round(float(scores[0]), 4),
        'features': row_data
    }


def predict_batch_dataframe(model, df):
    """
    Perform batch inference on a user-provided DataFrame.
    
    Returns:
        annotated_df: pd.DataFrame with 'prediction', 'anomaly_code', 'anomaly_score'
        summary: dict of metrics
    """
    valid, msg, clean_X = validate_input_data(df)
    if not valid:
        raise ValueError(msg)

    preds, labels, scores = predict_isolation_forest(model, clean_X)
    
    result_df = df.copy()
    result_df['anomaly_code'] = preds
    result_df['prediction'] = labels
    result_df['anomaly_score'] = scores.round(4)
    
    n_total = len(preds)
    n_anomalies = int((preds == -1).sum())
    n_normal = n_total - n_anomalies
    pct_anomaly = (n_anomalies / n_total) * 100 if n_total > 0 else 0.0

    summary = {
        'total_samples': n_total,
        'normal_count': n_normal,
        'anomaly_count': n_anomalies,
        'anomaly_percentage': round(pct_anomaly, 4)
    }

    return result_df, summary
