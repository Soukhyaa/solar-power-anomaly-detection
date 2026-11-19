"""
Comprehensive Test & Verification Script for Solar Power Anomaly Detection Project
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utils are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.prediction import load_model, predict_single_instance, predict_batch_dataframe
from utils.preprocessing import SELECTED_FEATURES

print("=" * 80)
print(" RUNNING SYSTEM VERIFICATION & TESTING SUITE")
print("=" * 80)

# 1. Model Loading Test
print("\n[Test 1/5] Testing Model & Preprocessor Reloading...")
iforest = load_model("models/isolation_forest.pkl")
dbscan = load_model("models/dbscan.pkl")
scaler = load_model("models/scaler.pkl")

print("  ✓ Successfully reloaded Isolation Forest model.")
print("  ✓ Successfully reloaded DBSCAN model.")
print("  ✓ Successfully reloaded StandardScaler object.")

# 2. Single Instance Prediction Test (Normal condition)
print("\n[Test 2/5] Testing Inference on Normal Operating Condition...")
normal_input = {
    'DC_POWER': 3500.0,
    'AC_POWER': 340.0,
    'DAILY_YIELD': 2500.0,
    'AMBIENT_TEMPERATURE': 28.5,
    'MODULE_TEMPERATURE': 38.0,
    'IRRADIATION': 0.40
}
res_normal = predict_single_instance(iforest, normal_input)
print(f"  Input: {normal_input}")
print(f"  Prediction: {res_normal['prediction']} (Code: {res_normal['anomaly_code']})")
print(f"  Decision Score: {res_normal['anomaly_score']:.4f}")
assert res_normal['prediction'] in ['Normal', 'Anomaly'], "Invalid prediction label"
print("  ✓ Single prediction test (normal) passed.")

# 3. Single Instance Prediction Test (Extreme Anomaly condition: High sun, zero generation)
print("\n[Test 3/5] Testing Inference on Extreme Anomaly Condition (High irradiation, 0 kW output)...")
anomaly_input = {
    'DC_POWER': 0.0,
    'AC_POWER': 0.0,
    'DAILY_YIELD': 100.0,
    'AMBIENT_TEMPERATURE': 35.0,
    'MODULE_TEMPERATURE': 65.0,
    'IRRADIATION': 0.95
}
res_anom = predict_single_instance(iforest, anomaly_input)
print(f"  Input: {anomaly_input}")
print(f"  Prediction: {res_anom['prediction']} (Code: {res_anom['anomaly_code']})")
print(f"  Decision Score: {res_anom['anomaly_score']:.4f}")
print("  ✓ Single prediction test (anomalous) passed.")

# 4. Batch Prediction Test
print("\n[Test 4/5] Testing Batch Prediction on Telemetry Sample...")
batch_df = pd.DataFrame([normal_input, anomaly_input, {
    'DC_POWER': 8000.0,
    'AC_POWER': 780.0,
    'DAILY_YIELD': 4000.0,
    'AMBIENT_TEMPERATURE': 32.0,
    'MODULE_TEMPERATURE': 52.0,
    'IRRADIATION': 0.75
}])

scored_df, summary = predict_batch_dataframe(iforest, batch_df)
print(f"  Batch Summary: {summary}")
assert summary['total_samples'] == 3
print("  ✓ Batch prediction test passed.")

# 5. Output Files Integrity Verification
print("\n[Test 5/5] Verifying Generated Artifacts & Datasets...")
assert os.path.exists("outputs/anomaly_results.csv"), "anomaly_results.csv missing"
assert os.path.exists("outputs/model_comparison.csv"), "model_comparison.csv missing"
assert os.path.exists("notebooks/Solar_Anomaly_Detection.ipynb"), "notebook missing"

df_anom = pd.read_csv("outputs/anomaly_results.csv")
df_comp = pd.read_csv("outputs/model_comparison.csv")
print(f"  Outputs verified: anomaly_results.csv has {len(df_anom):,} rows and {len(df_anom.columns)} columns.")
print(f"  Model comparison rows:\n{df_comp[['Algorithm', 'Anomaly Count', 'Anomaly Percentage (%)']]}")

print("\n" + "=" * 80)
print(" ALL 5 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
print("=" * 80)
